import runpod
from runpod.serverless.utils import rp_upload
import json
import logging
import urllib.request
import urllib.parse
import time
import os
import requests
import base64
from io import BytesIO
from toolkit.job import get_job

REFRESH_WORKER = os.environ.get("REFRESH_WORKER", "false").lower() == "true"
SAVE_MODEL_TO_FS_PATH = os.environ.get("SAVE_MODEL_TO_FS_PATH", '/runpod-volume/models/loras')

'''
# def validate_input(job_input):
#     """
#     Validates the input for the handler function.

#     Args:
#         job_input (dict): The input data to validate.

#     Returns:
#         tuple: A tuple containing the validated data and an error message, if any.
#                The structure is (validated_data, error_message).
#     """
#     # Validate if job_input is provided
#     if job_input is None:
#         return None, "Please provide input"

#     # Check if input is a string and try to parse it as JSON
#     if isinstance(job_input, str):
#         try:
#             job_input = json.loads(job_input)
#         except json.JSONDecodeError:
#             return None, "Invalid JSON format in input"

#     # Validate 'workflow' in input
#     workflow = job_input.get("workflow")
#     if workflow is None:
#         return None, "Missing 'workflow' parameter"

#     # Validate 'images' in input, if provided
#     images = job_input.get("images")
#     if images is not None:
#         if not isinstance(images, list) or not all(
#             "name" in image and "image" in image for image in images
#         ):
#             return (
#                 None,
#                 "'images' must be a list of objects with 'name' and 'image' keys",
#             )

#     # Return validated data and no error
#     return {"workflow": workflow, "images": images}, None
'''

from collections import OrderedDict
import os
import requests
from io import BytesIO
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import uuid
import shutil
import json

from toolkit.job import get_job

def run_captioning(images, concept_sentence, *captions):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        "multimodalart/Florence-2-large-no-flash-attn", torch_dtype=torch_dtype, trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained("multimodalart/Florence-2-large-no-flash-attn", trust_remote_code=True)

    captions = list(captions)
    for i, image_url in enumerate(images):
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_url}: {e}")
            captions[i] = "Error loading image"
            continue

        prompt = "<DETAILED_CAPTION>"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

        generated_ids = model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3
        )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text, task=prompt, image_size=(image.width, image.height)
        )
        caption_text = parsed_answer["<DETAILED_CAPTION>"].replace("The image shows ", "")
        if concept_sentence:
            caption_text = f"{caption_text} [trigger]"
        captions[i] = caption_text

    model.to("cpu")
    del model
    del processor

    return captions

def create_dataset(images, captions):
    logging.info("Creating dataset")
    destination_folder = f"datasets/{uuid.uuid4()}"
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        logging.debug(f"Created directory: {destination_folder}")

    jsonl_file_path = os.path.join(destination_folder, "metadata.jsonl")
    with open(jsonl_file_path, "a") as jsonl_file:
        for index, image_url in enumerate(images):
            try:
                response = requests.get(image_url)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
                local_image_path = os.path.join(destination_folder, f"image_{index}.jpg")
                image.save(local_image_path)
                logging.info(f"Saved image {local_image_path}")
            except Exception as e:
                logging.error(f"Error saving image {image_url}: {e}")
                continue

            original_caption = captions[index]
            file_name = os.path.basename(local_image_path)

            data = {"file_name": file_name, "prompt": original_caption}
            jsonl_file.write(json.dumps(data) + "\n")
            logging.debug(f"Written metadata for {file_name}")

    logging.info(f"Dataset created at: {destination_folder}")
    return destination_folder

def create_captioned_dataset(image_urls, concept_sentence, *captions):
    final_captions = run_captioning(image_urls, concept_sentence, *captions)
    return create_dataset(image_urls, final_captions)

def get_config(name: str, dataset_dir: str, output_dir: str, steps: int = 1000, seed: int = 42):
    # TODO: allow passing config as job input
    return OrderedDict([
    ('job', 'extension'),
    ('config', OrderedDict([
        # this name will be the folder and filename name
        ('name', name),
        ('process', [
            OrderedDict([
                ('type', 'sd_trainer'),
                # root folder to save training sessions/samples/weights
                ('training_folder', output_dir),
                # uncomment to see performance stats in the terminal every N steps
                ('performance_log_every', 1000),
                ('device', 'cuda:0'),
                # if a trigger word is specified, it will be added to captions of training data if it does not already exist
                # alternatively, in your captions you can add [trigger] and it will be replaced with the trigger word
                # ('trigger_word', 'image'),
                ('network', OrderedDict([
                    ('type', 'lora'),
                    ('linear', 16),
                    ('linear_alpha', 16)
                ])),
                # TODO: this pauses the process every N steps, do we need this?
                # ('save', OrderedDict([
                #     ('dtype', 'float16'),  # precision to save
                #     ('save_every', 250),  # save every this many steps
                #     ('max_step_saves_to_keep', 4)  # how many intermittent saves to keep
                # ])),
                ('datasets', [
                    # datasets are a folder of images. captions need to be txt files with the same name as the image
                    # for instance image2.jpg and image2.txt. Only jpg, jpeg, and png are supported currently
                    # images will automatically be resized and bucketed into the resolution specified
                    OrderedDict([
                        ('folder_path', dataset_dir),
                        ('caption_ext', 'txt'),
                        ('caption_dropout_rate', 0.05),  # will drop out the caption 5% of time
                        ('shuffle_tokens', False),  # shuffle caption order, split by commas
                        ('cache_latents_to_disk', True),  # leave this true unless you know what you're doing
                        ('resolution', [512, 768, 1024])  # flux enjoys multiple resolutions
                    ])
                ]),
                ('train', OrderedDict([
                    ('batch_size', 1),
                    ('steps', steps),  # total number of steps to train 500 - 4000 is a good range
                    ('gradient_accumulation_steps', 1),
                    ('train_unet', True),
                    ('train_text_encoder', False),  # probably won't work with flux
                    ('content_or_style', 'balanced'),  # content, style, balanced
                    ('gradient_checkpointing', True),  # need the on unless you have a ton of vram
                    ('noise_scheduler', 'flowmatch'),  # for training only
                    ('optimizer', 'adamw8bit'),
                    ('lr', 1e-4),

                    # uncomment this to skip the pre training sample
                    # ('skip_first_sample', True),

                    # uncomment to completely disable sampling
                    # ('disable_sampling', True),

                    # uncomment to use new vell curved weighting. Experimental but may produce better results
                    # ('linear_timesteps', True),

                    # ema will smooth out learning, but could slow it down. Recommended to leave on.
                    ('ema_config', OrderedDict([
                        ('use_ema', True),
                        ('ema_decay', 0.99)
                    ])),

                    # will probably need this if gpu supports it for flux, other dtypes may not work correctly
                    ('dtype', 'bf16')
                ])),
                ('model', OrderedDict([
                    # huggingface model name or path
                    ('name_or_path', 'black-forest-labs/FLUX.1-dev'),
                    ('is_flux', True),
                    ('quantize', True),  # run 8bit mixed precision
                    #('low_vram', True),  # uncomment this if the GPU is connected to your monitors. It will use less vram to quantize, but is slower.
                ])),
                ('sample', OrderedDict([
                    ('sampler', 'flowmatch'),  # must match train.noise_scheduler
                    # TODO: persist sampling results
                    ('sample_every', steps),  # sample every this many steps
                    ('width', 1024),
                    ('height', 1024),
                    ('prompts', [
                        # you can add [trigger] to the prompts here and it will be replaced with the trigger word
                        #'[trigger] holding a sign that says \'I LOVE PROMPTS!\'',
                        'woman with red hair, playing chess at the park, bomb going off in the background',
                        'a woman holding a coffee cup, in a beanie, sitting at a cafe',
                        'a horse is a DJ at a night club, fish eye lens, smoke machine, lazer lights, holding a martini',
                        'a man showing off his cool new t shirt at the beach, a shark is jumping out of the water in the background',
                        'a bear building a log cabin in the snow covered mountains',
                        'woman playing the guitar, on stage, singing a song, laser lights, punk rocker',
                        'hipster man with a beard, building a chair, in a wood shop',
                        'photo of a man, white background, medium shot, modeling clothing, studio lighting, white backdrop',
                        'a man holding a sign that says, \'this is a sign\'',
                        'a bulldog, in a post apocalyptic world, with a shotgun, in a leather jacket, in a desert, with a motorcycle'
                    ]),
                    ('neg', ''),  # not used on flux
                    ('seed', seed),
                    ('walk_seed', True),
                    ('guidance_scale', 4),
                    ('sample_steps', 20)
                ]))
            ])
        ])
    ])),
    # you can add any additional meta info here. [name] is replaced with config name at top
    ('meta', OrderedDict([
        ('name', '[name]'),
        ('version', '1.0')
    ]))
])

# Run the captioning
# final_captions = run_captioning(image_urls, concept_sentence, *captions)
# print(final_captions)

# # Create the dataset
# dataset_folder = create_dataset(image_urls, final_captions)


def handler(job):
    """
    # TODO: docs
    The main function that handles a job  an image.

    This function validates the input, sends a prompt to ComfyUI for processing,
    polls ComfyUI for result, and retrieves generated images.

    Args:
        job (dict): A dictionary containing job details and input parameters.

    Returns:
        dict: A dictionary containing either an error message or a success status with generated images.
    """
    # TODO: validate
    # validated_data, error_message = validate_input(job_input)
    # if error_message:
    #     return {"error": error_message}
    
    job_input = job["input"]
    image_urls = job_input["images"]
    name = job_input["name"]
    concept_sentence = job_input.get("concept_sentence", "")

    # image_urls = [
    #     "https://images.pexels.com/photos/774909/pexels-photo-774909.jpeg",
    #     "https://images.pexels.com/photos/1239291/pexels-photo-1239291.jpeg"
    # ]
    captions = [""] * len(image_urls)

    dataset_folder = create_captioned_dataset(image_urls, concept_sentence, *captions)
    config = get_config(name, dataset_folder, SAVE_MODEL_TO_FS_PATH)
    job = get_job(config, name)
    job.run()
    job.cleanup()

    # Make sure that the input is valid

    # # Extract validated data
    # workflow = validated_data["workflow"]
    # images = validated_data.get("images")

    result = {"result": "test", "refresh_worker": REFRESH_WORKER}

    return result

# Start the handler only if this script is run directly
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
