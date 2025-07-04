from typing import Literal
import runpod
import json
import logging
import os
import requests
import subprocess
import yaml
from io import BytesIO
from loki_logger_handler.loki_logger_handler import LokiLoggerHandler
import sys
from collections import OrderedDict
from PIL import Image, ImageOps
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import uuid

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

LOKI_URL = os.getenv("LOKI_URL")

if LOKI_URL:
    logger.info("Configuring Loki logging.")
    loki_handler = LokiLoggerHandler(
        url=LOKI_URL,
        labels={"app": "flux-app-training-serverless-worker"}
    )
    logger.addHandler(loki_handler)
else:
    logger.warning("Loki credentials not provided, falling back to local logging.")

    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_record = {
                'time': self.formatTime(record),
                'name': record.name,
                'level': record.levelname,
            }

            for attr in vars(record):
                if attr not in log_record:
                    value = getattr(record, attr)
                    # Convert non-JSON-serializable objects to strings
                    try:
                        json.dumps(value)  # Test if it's JSON serializable
                        log_record[attr] = value
                    except (TypeError, ValueError):
                        # Convert to string if not JSON serializable
                        log_record[attr] = str(value)

            return json.dumps(log_record)

    local_handler = logging.StreamHandler(sys.stdout)
    local_handler.setLevel(logging.DEBUG)
    local_handler.setFormatter(JSONFormatter())

    logger.addHandler(local_handler)

REFRESH_WORKER = os.environ.get("REFRESH_WORKER", "false").lower() == "true"
SAVE_MODEL_TO_FS_PATH = os.environ.get("SAVE_MODEL_TO_FS_PATH", '/runpod-volume/models/loras')
IMAGE_QUALITY = int(os.environ.get("IMAGE_QUALITY", 95))

class DictFilter(logging.Filter):
    def __init__(self, extra_dict=None):
        super().__init__()
        self.extra_dict = extra_dict or {}

    def filter(self, record):
        # Assign each key-value pair in the dictionary to the log record
        for key, value in self.extra_dict.items():
            setattr(record, key, value)
        return True

def handler(job):
    """
    The main function that handles a job for image processing.

    This function validates the input, sends a prompt to ComfyUI for processing,
    polls ComfyUI for result, and retrieves generated images.

    Args:
        job (dict): A dictionary containing job details and input parameters.

    Returns:
        dict: A dictionary containing either an error message or a success status with generated images.
    """
    try:
        # Input validation
        if not isinstance(job, dict):
            return {"error": "Job must be a dictionary"}
        
        if "id" not in job:
            return {"error": "Job must contain an 'id' field"}
        
        if "input" not in job:
            return {"error": "Job must contain an 'input' field"}
        
        logger.addFilter(DictFilter({"runpod_request_id": job["id"]}))
        logger.debug("Got job", extra={"job": job, "test": "bar"})
        
        job_input = job["input"]
        
        # Validate job_input structure
        if not isinstance(job_input, dict):
            return {"error": "Job input must be a dictionary"}
        
        # Validate required fields
        required_fields = ["images", "name", "gender"]
        for field in required_fields:
            if field not in job_input:
                return {"error": f"Missing required field: {field}"}
        
        # Validate images
        image_urls = job_input["images"]
        if not isinstance(image_urls, list) or len(image_urls) == 0:
            return {"error": "images must be a non-empty list of URLs"}
        
        for url in image_urls:
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                return {"error": f"Invalid image URL: {url}"}
        
        # Validate name
        name = job_input["name"]
        if not isinstance(name, str) or not name.strip():
            return {"error": "name must be a non-empty string"}
        
        # Process valid input
        dataset_folder, captions = create_captioned_dataset(image_urls, False)
        config = get_config(name, dataset_folder, SAVE_MODEL_TO_FS_PATH)
        
        # Save config as YAML in the dataset folder
        config_path = os.path.join(dataset_folder, "config.yaml")
        
        # Convert OrderedDict to regular dict recursively
        def convert_ordered_dict_to_dict(ordered_dict):
            if isinstance(ordered_dict, OrderedDict):
                return {k: convert_ordered_dict_to_dict(v) for k, v in ordered_dict.items()}
            elif isinstance(ordered_dict, list):
                return [convert_ordered_dict_to_dict(item) for item in ordered_dict]
            else:
                return ordered_dict
        
        # Convert the config to regular dict before saving
        regular_dict_config = convert_ordered_dict_to_dict(config)
        
        with open(config_path, 'w') as f:
            yaml.dump(regular_dict_config, f, sort_keys=False)
        
        logger.info(
            "Saved config",
            extra={
                "job_name": name,
                "config_path": config_path,
                "captions": captions,
                "dataset_folder_path": dataset_folder,
                "dataset_folder_relative_path": os.path.abspath(dataset_folder)
            }
        )
        
        # Run the training script using the saved config
        ai_toolkit_path = "/workspace/ai-toolkit"  # Path to the ai-toolkit repo
        run_script_path = os.path.join(ai_toolkit_path, "run.py")
        
        cmd = ["python", run_script_path, config_path]
        
        logger.debug("Clearing cuda cache")
        torch.cuda.empty_cache()
        logger.info("Running command", extra={"cmd": " ".join(cmd)})
        
        # Use Popen to capture output in real-time
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout to capture both
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Stream and log output in real-time
        output_lines = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                output_line = output.strip()
                output_lines.append(output_line)
                logger.info(output_line, extra={"subprocess": True})
        
        # Wait for process to complete and get return code
        return_code = process.poll()
        
        if return_code != 0:
            full_output = '\n'.join(output_lines)
            logger.error("Command failed", extra={
                "returncode": return_code,
                "output": full_output
            })
            raise Exception(f"Training failed with exit code {return_code}")
        
        full_output = '\n'.join(output_lines)
        logger.info("Command completed successfully", extra={"output": full_output})

        files_to_delete = ['config.yaml', 'optimizer.pt']
        logger.info("Cleaning up files", extra={"files": files_to_delete})

        for file_name in files_to_delete:
            file_path = os.path.join(SAVE_MODEL_TO_FS_PATH, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
                logger.info("Deleted file", extra={"file_path": file_path})
            else:
                logger.warning("File not found", extra={"file_path": file_path})

        result = {"result": dataset_folder, "refresh_worker": REFRESH_WORKER}
        return result

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            log_gpu_memory_usage()
            log_memory_usage()
            logger.error("CUDA out of memory error", exc_info=True)
            return {"error": "CUDA out of memory"}
        else:
            logger.error("Job errored", exc_info=True)
            return {"error": str(e)}
    except Exception as e:
        logger.error("Job errored", exc_info=True)
        return {"error": str(e)}

def log_memory_usage():
    allocated_memory = torch.cuda.memory_allocated()
    reserved_memory = torch.cuda.memory_reserved()
    logger.debug(f"Allocated memory: {allocated_memory / (1024 ** 2):.2f} MiB")
    logger.debug(f"Reserved memory: {reserved_memory / (1024 ** 2):.2f} MiB")
    
def log_gpu_memory_usage():
    try:
        # Run nvidia-smi to get GPU memory usage
        process = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_memory_usage", "--format=csv"],
            capture_output=True,
            text=True
        )
        if process.returncode == 0:
            logger.debug("Current GPU memory usage:\n" + process.stdout)
        else:
            logger.error(f"Failed to run nvidia-smi: {process.stderr}")
    except Exception as e:
        logger.error("Error logging GPU memory usage", exc_info=True)

def generate_caption_for_image(image, processor, model, device, torch_dtype, concept_sentence=False):
    """
    Generate a caption for a single image using the Florence model.
    
    Args:
        image (PIL.Image): The image to caption
        processor: The text processor
        model: The captioning model
        device: The device to run inference on
        torch_dtype: The torch data type
        concept_sentence (bool): Whether to add a trigger token
        
    Returns:
        str: The generated caption
    """
    prompt = "<DETAILED_CAPTION>"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, task=prompt, image_size=(image.width, image.height)
    )
    caption_text = clean_caption(parsed_answer["<DETAILED_CAPTION>"])
    if concept_sentence:
        caption_text = f"{caption_text} [trigger]"
    
    return caption_text

def clean_caption(cap, replacements=None):
    default_replacements = [
        ("the image features", ""),
        ("the image shows", ""),
        ("the image depicts", ""),
        ("the image is", ""),
        ("in this image", ""),
        ("in the image", ""),
        ("gigi hadid", "a woman"),
    ]
        
    if replacements is None:
        replacements = default_replacements

    # remove any newlines
    cap = cap.replace("\n", ", ")
    cap = cap.replace("\r", ", ")
    cap = cap.replace(".", ",")
    cap = cap.replace("\"", "")

    # remove unicode characters
    cap = cap.encode('ascii', 'ignore').decode('ascii')

    # make lowercase
    cap = cap.lower()
    # remove any extra spaces
    cap = " ".join(cap.split())

    for replacement in replacements:
        if replacement[0].startswith('*'):
            # we are removing all text if it starts with this and the rest matches
            search_text = replacement[0][1:]
            if cap.startswith(search_text):
                cap = ""
        else:
            cap = cap.replace(replacement[0].lower(), replacement[1].lower())

    cap_list = cap.split(",")
    # trim whitespace
    cap_list = [c.strip() for c in cap_list]
    # remove empty strings
    cap_list = [c for c in cap_list if c != ""]
    # remove duplicates
    cap_list = list(dict.fromkeys(cap_list))
    # join back together
    cap = ", ".join(cap_list)
    return cap

def run_captioning(images, concept_sentence):
    """
    Generate captions for a list of images using the Florence-2 model.
    
    Args:
        images (list): List of image URLs
        concept_sentence (str): Concept sentence to append to captions if needed
    
    Returns:
        list: Generated captions for each image
    """
    logger.info("Starting image captioning process", extra={"num_images": len(images)})
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug(f"Using device", extra={"device": device})
    
    model_name = 'microsoft/Florence-2-large'
    torch_dtype = torch.float16
    
    logger.info("Loading Florence-2 model and processor", extra={"model_name": model_name})
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch_dtype, 
            trust_remote_code=True
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        logger.debug("Model and processor loaded successfully")
    except Exception as e:
        logger.error("Failed to load model or processor", extra={"error": str(e), "model_name": model_name})
        raise

    captions = [""] * len(images)  # Create captions list
    for i, image_url in enumerate(images):
        logger.info(f"Processing image", extra={"image_url": image_url, "n": i+1, "count": len(images)})
        try:
            logger.debug("Downloading image", extra={"image_url": image_url})
            response = requests.get(image_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            logger.debug("Image downloaded and converted successfully", extra={"image_url": image_url, "image_size": image.size})
        except Exception as e:
            logger.error("Error loading image", extra={"image_url": image_url, "error": str(e)})
            captions[i] = ""
            continue

        logger.debug("Generating caption for image", extra={"image_index": i})
        captions[i] = generate_caption_for_image(image, processor, model, device, torch_dtype, concept_sentence)
        logger.info("Caption generated successfully", extra={"image_index": i, "caption_length": len(captions[i]), "caption": captions[i]})

    logger.info("Captioning process completed", extra={"num_images": len(images)})
    logger.debug("Cleaning up model resources")
    del model
    del processor
    if device == "cuda":
        torch.cuda.empty_cache()
        logger.debug("CUDA cache cleared")

    return captions

def create_dataset(images, captions):
    logger.info("Creating dataset")
    destination_folder = f"datasets/{uuid.uuid4()}"
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        logger.debug("Created directory", extra={"directory": destination_folder})

    for index, image_url in enumerate(images):
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            local_image_path = os.path.join(destination_folder, f"image_{index}.jpg")
            image.save(local_image_path, quality=IMAGE_QUALITY)
            logger.debug("Saved image", extra={"path": os.path.abspath(local_image_path), "relative_path": local_image_path})
        except Exception as e:
            logger.error("Error saving image", extra={"image_url": image_url, "error": str(e)})
            continue

        original_caption = captions[index]
        
        # Save the caption to a text file with the same name as the image
        caption_file_path = os.path.join(destination_folder, f"image_{index}.txt")
        with open(caption_file_path, "w") as caption_file:
            caption_file.write(original_caption)
            logger.debug("Saved caption", extra={"path": os.path.abspath(caption_file_path), "relative_path": caption_file_path})

    logger.info("Dataset created", extra={"path": os.path.abspath(destination_folder), "relative_path": destination_folder})
    return destination_folder

def create_captioned_dataset(image_urls, concept_sentence):
    captions = run_captioning(image_urls, concept_sentence)
    dataset_folder = create_dataset(image_urls, captions)
    return (dataset_folder, captions)

def get_config(name: str, dataset_dir: str, output_dir: str, steps: int = 1500, seed: int = 42):
    # example workflow https://github.com/ostris/ai-toolkit/blob/main/config/examples/train_lora_flux_24gb.yaml
    # TODO: allow passing config as job input
    # TODO: we are saving some `optimizer.pt` along with `.safetensors` - this is not needed, research how to remove
    
    return OrderedDict([
        ('job', 'extension'),
        ('config', OrderedDict([
            # this name will be the folder and filename name
            ('name', name),
            ('process', [
                OrderedDict([
                    ('type', 'sd_trainer'),
                    ('training_folder', output_dir),
                    ('performance_log_every', 1000),
                    ('device', 'cuda:0'),
                    ('network', OrderedDict([
                        ('type', 'lora'),
                        ('linear', 16),
                        ('linear_alpha', 16),
                        ('lokr_full_rank', True),
                        ('lokr_factor', -1)
                    ])),
                    ('save', OrderedDict([
                        ('dtype', 'bf16'),
                        ('save_every', 1500),
                        ('max_step_saves_to_keep', 1),
                        ('save_format', 'diffusers'),
                        ('push_to_hub', False)
                    ])),
                    ('datasets', [
                        OrderedDict([
                            ('folder_path', dataset_dir),
                            ('mask_path', None),
                            ('mask_min_value', 0.1),
                            ('default_caption', ''),
                            ('caption_ext', 'txt'),
                            ('caption_dropout_rate', 0.05),
                            ('cache_latents_to_disk', False),
                            ('is_reg', False),
                            ('network_weight', 1),
                            ('resolution', [512, 768, 1024])
                        ])
                    ]),
                    ('train', OrderedDict([
                        ('batch_size', 1),
                        ('bypass_guidance_embedding', False),
                        ('steps', steps),
                        ('gradient_accumulation', 1),
                        ('train_unet', True),
                        ('train_text_encoder', False),
                        ('gradient_checkpointing', True),
                        ('noise_scheduler', 'flowmatch'),
                        ('optimizer', 'adamw8bit'),
                        ('timestep_type', 'sigmoid'),
                        ('content_or_style', 'balanced'),
                        ('optimizer_params', OrderedDict([
                            ('weight_decay', 0.0001)
                        ])),
                        ('unload_text_encoder', False),
                        ('lr', 0.0001),
                        ('ema_config', OrderedDict([
                            ('use_ema', True),
                            ('ema_decay', 0.99)
                        ])),
                        ('dtype', 'bf16'),
                        ('diff_output_preservation', False),
                        ('diff_output_preservation_multiplier', 1),
                        ('diff_output_preservation_class', 'person'),
                        ('skip_first_sample', True),
                        ('disable_sampling', True),
                    ])),
                    ('model', OrderedDict([
                        ('name_or_path', '/runpod-volume/flux'),
                        ('quantize', True),
                        ('quantize_te', True),
                        ('arch', 'flux'),
                        ('low_vram', True)
                    ])),
                    ('sample', OrderedDict([
                        ('sampler', 'flowmatch'),
                        ('sample_every', 250),
                        ('width', 1024),
                        ('height', 1024),
                        ('prompts', ['woman with red hair, playing chess at the park, bomb going off in the background']),
                        ('neg', ''),
                        ('seed', seed),
                        ('walk_seed', True),
                        ('guidance_scale', 4),
                        ('sample_steps', 25),
                        ('num_frames', 1),
                        ('fps', 1)
                    ]))
                ])
            ])
        ])),
        ('meta', OrderedDict([
            ('name', '[name]'),
            ('version', '1.0')
        ]))
    ])

# Start the handler only if this script is run directly
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
