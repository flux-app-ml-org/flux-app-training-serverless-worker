# tests/test_rp_handler.py
import sys
import pytest
from unittest.mock import patch, MagicMock, mock_open, ANY
import json
import os
import shutil
import tempfile
import yaml
from io import BytesIO
from PIL import Image as PILImage  # We'll use the real PIL for image operations

# Mock heavy dependencies before importing the module under test
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['runpod'] = MagicMock()
sys.modules['loki_logger_handler'] = MagicMock()
sys.modules['loki_logger_handler.loki_logger_handler'] = MagicMock()

# Now we can safely import the module
import rp_handler

class MockResponse:
    def __init__(self, content=None, status_code=200):
        # Create a small test image if content is None
        if content is None:
            img = PILImage.new('RGB', (100, 100), color='red')
            buffer = BytesIO()
            img.save(buffer, format='JPEG')
            self.content = buffer.getvalue()
        else:
            self.content = content
        self.status_code = status_code
    
    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP Error: {self.status_code}")

@pytest.fixture
def temp_dir():
    # Create a temporary directory for test files
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up after the test
    shutil.rmtree(temp_dir)

# Constants for expected config values
EXPECTED_CONFIG_VALUES = {
    "job": "extension",
    "network_type": "lora",
    "network_linear": 16,
    "network_linear_alpha": 16,
    "caption_ext": "txt",
    "caption_dropout_rate": 0.05,
    "cache_latents_to_disk": False,
    "resolution": [512, 768, 1024],
    "batch_size": 1,
    "default_steps": 1000,
    "gradient_accumulation": 1,
    "train_unet": True,
    "train_text_encoder": False,
    "gradient_checkpointing": True,
    "noise_scheduler": "flowmatch",
    "optimizer": "adamw8bit",
    "lr": 1e-4,
    "disable_sampling": True,
    "ema_use": True,
    "ema_decay": 0.99,
    "dtype": "bf16",
    "model_name": "black-forest-labs/FLUX.1-dev",
    "arch": "flux",
    "quantize": True
}

def verify_config_structure(config_content):
    """Helper function to verify the structure of the config file"""
    # Check top-level keys
    assert config_content["job"] == EXPECTED_CONFIG_VALUES["job"]
    assert "config" in config_content
    assert "meta" in config_content
    
    # Check process section
    process = config_content["config"]["process"][0]
    assert process["type"] == "sd_trainer"
    assert process["device"] == "cuda:0"
    
    # Check network configuration
    assert process["network"]["type"] == EXPECTED_CONFIG_VALUES["network_type"]
    assert process["network"]["linear"] == EXPECTED_CONFIG_VALUES["network_linear"]
    assert process["network"]["linear_alpha"] == EXPECTED_CONFIG_VALUES["network_linear_alpha"]
    
    # Check dataset configuration
    dataset_config = process["datasets"][0]
    assert dataset_config["caption_ext"] == EXPECTED_CONFIG_VALUES["caption_ext"]
    assert dataset_config["caption_dropout_rate"] == EXPECTED_CONFIG_VALUES["caption_dropout_rate"]
    # Removed shuffle_tokens check as it's not in the function
    assert dataset_config["cache_latents_to_disk"] is EXPECTED_CONFIG_VALUES["cache_latents_to_disk"]
    assert dataset_config["resolution"] == EXPECTED_CONFIG_VALUES["resolution"]
    
    # Check training configuration
    train_config = process["train"]
    assert train_config["batch_size"] == EXPECTED_CONFIG_VALUES["batch_size"]
    assert train_config["gradient_accumulation"] == EXPECTED_CONFIG_VALUES["gradient_accumulation"]  # Updated key name
    assert train_config["train_unet"] is EXPECTED_CONFIG_VALUES["train_unet"]
    assert train_config["train_text_encoder"] is EXPECTED_CONFIG_VALUES["train_text_encoder"]
    assert train_config["gradient_checkpointing"] is EXPECTED_CONFIG_VALUES["gradient_checkpointing"]
    assert train_config["noise_scheduler"] == EXPECTED_CONFIG_VALUES["noise_scheduler"]
    assert train_config["optimizer"] == EXPECTED_CONFIG_VALUES["optimizer"]
    assert train_config["lr"] == EXPECTED_CONFIG_VALUES["lr"]
    assert train_config["disable_sampling"] is EXPECTED_CONFIG_VALUES["disable_sampling"]
    assert train_config["dtype"] == EXPECTED_CONFIG_VALUES["dtype"]
    
    # Check EMA configuration
    assert train_config["ema_config"]["use_ema"] is EXPECTED_CONFIG_VALUES["ema_use"]
    assert train_config["ema_config"]["ema_decay"] == EXPECTED_CONFIG_VALUES["ema_decay"]
    
    # Check model configuration
    model_config = process["model"]
    assert model_config["name_or_path"] == EXPECTED_CONFIG_VALUES["model_name"]
    assert model_config["arch"] == EXPECTED_CONFIG_VALUES["arch"]  # Changed from is_flux to arch
    assert model_config["quantize"] is EXPECTED_CONFIG_VALUES["quantize"]

@pytest.fixture
def setup_mocks(temp_dir):
    # Set up all the necessary mocks
    rp_handler.torch.cuda.is_available = MagicMock(return_value=True)
    rp_handler.torch.float16 = "float16"
    rp_handler.torch.cuda.empty_cache = MagicMock()
    
    # Mock subprocess.Popen to avoid actually running the command
    class MockProcess:
        def __init__(self):
            self.output_lines = ["Training output line 1", "Training output line 2"]
            self.current_line = 0
            self.return_code = 0
            
        def readline(self):
            if self.current_line < len(self.output_lines):
                line = self.output_lines[self.current_line] + "\n"
                self.current_line += 1
                return line
            return ""  # Return empty string when no more lines
            
        def poll(self):
            # Return None while there are still lines to read, then return the exit code
            if self.current_line < len(self.output_lines):
                return None
            return self.return_code
    
    mock_process = MockProcess()
    mock_process.stdout = mock_process  # stdout.readline() calls the readline method
    rp_handler.subprocess.Popen = MagicMock(return_value=mock_process)
    
    # Set environment variables
    os.environ["SAVE_MODEL_TO_FS_PATH"] = os.path.join(temp_dir, "models")
    
    # Create a fixed UUID for testing
    rp_handler.uuid.uuid4 = MagicMock(return_value="test-uuid")
    
    # Override dataset directory to use our temp dir
    original_create_dataset = rp_handler.create_dataset
    def mock_create_dataset(images, captions):
        dataset_dir = os.path.join(temp_dir, f"datasets/{rp_handler.uuid.uuid4()}")
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Actually save some files for testing
        for index, image_url in enumerate(images):
            # Create a test image file
            img = PILImage.new('RGB', (100, 100), color='red')
            local_image_path = os.path.join(dataset_dir, f"image_{index}.jpg")
            img.save(local_image_path)
            
            # Save the caption to a text file
            caption_file_path = os.path.join(dataset_dir, f"image_{index}.txt")
            with open(caption_file_path, "w") as caption_file:
                caption_file.write(captions[index])
        
        return dataset_dir
    
    rp_handler.create_dataset = mock_create_dataset
    
    yield {
        "temp_dir": temp_dir,
    }
    
    # Restore original functions
    rp_handler.create_dataset = original_create_dataset

@pytest.fixture
def mock_requests():
    with patch('requests.get') as mock_get:
        mock_get.return_value = MockResponse()
        yield mock_get

def test_handler_valid_input(setup_mocks, mock_requests):
    # Test with valid input
    job = {
        "id": "test-job-id",
        "input": {
            "images": ["https://example.com/image1.jpg", "https://example.com/image2.jpg"],
            "name": "test-model",
            "gender": "F"
        }
    }
    
    # Mock the run_captioning function to return predictable captions
    expected_captions = ["Expected caption for image 0", "Expected caption for image 1"]
    with patch('rp_handler.run_captioning', return_value=expected_captions):
        # Call the handler
        result = rp_handler.handler(job)
        
        # Verify the result
        assert "result" in result
        assert "refresh_worker" in result
        
        # Verify the dataset was created
        dataset_path = result["result"]
        assert os.path.exists(dataset_path)
        
        # Verify image files were created
        assert os.path.exists(os.path.join(dataset_path, "image_0.jpg"))
        assert os.path.exists(os.path.join(dataset_path, "image_1.jpg"))
        
        # Verify caption files were created and contain expected content
        for i, expected_caption in enumerate(expected_captions):
            caption_file_path = os.path.join(dataset_path, f"image_{i}.txt")
            assert os.path.exists(caption_file_path)
            
            # Verify the content of the caption file
            with open(caption_file_path, "r") as f:
                file_content = f.read()
                assert file_content == expected_caption
        
        # Verify config.yaml was created and contains the expected content
        config_path = os.path.join(dataset_path, "config.yaml")
        assert os.path.exists(config_path)
        
        with open(config_path, "r") as f:
            config_content = yaml.safe_load(f)
            
            # Check config name
            assert config_content["config"]["name"] == "test-model"
            
            # Check specific paths
            process = config_content["config"]["process"][0]
            assert process["training_folder"] == "/runpod-volume/models/loras"
            assert process["datasets"][0]["folder_path"] == dataset_path
            
            # Verify the rest of the config structure using the helper function
            verify_config_structure(config_content)
            
            # Check steps value specifically
            assert process["train"]["steps"] == EXPECTED_CONFIG_VALUES["default_steps"]
        
        # Verify subprocess.Popen was called with the correct parameters
        rp_handler.subprocess.Popen.assert_called_once()
        cmd_args = rp_handler.subprocess.Popen.call_args[0][0]
        assert cmd_args[0] == "python"
        assert cmd_args[1].endswith("run.py")
        assert cmd_args[2] == config_path

def test_create_captioned_dataset_content(setup_mocks, mock_requests):
    """Test that the captions are correctly written to text files"""
    image_urls = ["https://example.com/image1.jpg", "https://example.com/image2.jpg"]
    
    # Create specific test captions
    test_captions = ["Test caption for image 0", "Test caption for image 1"]
    
    # Mock the run_captioning function to return our test captions
    with patch('rp_handler.run_captioning', return_value=test_captions):
        # Call the function
        dataset_folder, captions = rp_handler.create_captioned_dataset(image_urls, False)
        
        # Verify the dataset was created
        assert os.path.exists(dataset_folder)
        
        # Verify the content of the caption files matches our test captions
        for i, caption in enumerate(test_captions):
            caption_file_path = os.path.join(dataset_folder, f"image_{i}.txt")
            assert os.path.exists(caption_file_path)
            
            with open(caption_file_path, "r") as f:
                file_content = f.read()
                assert file_content == caption
        
@pytest.fixture
def mock_cuda():
    with patch('torch.cuda') as mock_cuda:
        mock_cuda.is_available.return_value = True
        mock_cuda.empty_cache = MagicMock()
        yield mock_cuda

@pytest.fixture
def mock_image_generation():
    with patch('PIL.Image.open') as mock_open:
        # Create a real test image for consistent testing
        test_image = PILImage.new('RGB', (100, 100), color='red')
        mock_open.return_value = test_image
        yield test_image

def test_generate_caption_for_image(mock_cuda, mock_image_generation):
    """Test the single image captioning function"""
    test_image = PILImage.new('RGB', (100, 100), color='red')
    
    # Create minimal mocks for processor and model
    mock_processor = MagicMock()
    mock_model = MagicMock()
    
    # Set up the expected behavior
    expected_caption = "a beautiful sunset over mountains"
    mock_processor.post_process_generation.return_value = {
        "<DETAILED_CAPTION>": f"The image shows {expected_caption}"
    }
    
    # Test without concept_sentence
    caption = rp_handler.generate_caption_for_image(
        test_image, mock_processor, mock_model, "cuda", "float16", False
    )
    assert caption == expected_caption
    
    # Test with concept_sentence
    caption_with_trigger = rp_handler.generate_caption_for_image(
        test_image, mock_processor, mock_model, "cuda", "float16", True
    )
    assert caption_with_trigger == f"{expected_caption} [trigger]"
    
    # Verify processor and model were called correctly
    mock_processor.assert_called_with(
        text="<DETAILED_CAPTION>", 
        images=test_image, 
        return_tensors="pt"
    )
    mock_model.generate.assert_called()

def test_run_captioning_end_to_end(mock_cuda, mock_image_generation, temp_dir):
    """Test the full captioning pipeline with mocked model components"""
    # Setup test data
    image_urls = ["https://example.com/image1.jpg", "https://example.com/image2.jpg"]
    
    # Mock the model and processor
    with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model_cls, \
         patch('transformers.AutoProcessor.from_pretrained') as mock_processor_cls, \
         patch('requests.get') as mock_get:
        
        # Setup mock responses
        mock_get.return_value = MockResponse()
        
        # Setup model and processor behavior
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_model_cls.return_value = mock_model
        mock_processor_cls.return_value = mock_processor
        
        # Configure processor to return specific captions
        mock_processor.post_process_generation.side_effect = [
            {"<DETAILED_CAPTION>": "The image shows a person walking"},
            {"<DETAILED_CAPTION>": "The image shows a sunset"}
        ]
        
        # Run the captioning
        result_captions = rp_handler.run_captioning(image_urls, False)
        
        # Verify results
        assert len(result_captions) == 2
        assert result_captions[0] == "a person walking"
        assert result_captions[1] == "a sunset"
        
        # Verify model and processor were used correctly
        assert mock_model_cls.call_count == 1
        assert mock_processor_cls.call_count == 1
        assert mock_processor.post_process_generation.call_count == 2
        
        # Verify CUDA usage
        mock_cuda.empty_cache.assert_called_once()

def test_create_dataset_structure(temp_dir, mock_image_generation):
    """Test that dataset is created with correct structure"""
    images = ["https://example.com/image1.jpg", "https://example.com/image2.jpg"]
    captions = ["Test caption 1", "Test caption 2"]
    
    with patch('requests.get') as mock_get:
        mock_get.return_value = MockResponse()
        
        # Create dataset
        dataset_folder = rp_handler.create_dataset(images, captions)
        
        # Verify structure
        assert os.path.exists(dataset_folder)
        
        # Check image files
        for i in range(2):
            image_path = os.path.join(dataset_folder, f"image_{i}.jpg")
            assert os.path.exists(image_path)
            
            # Verify image content
            img = PILImage.open(image_path)
            assert img.size == (100, 100)
            
            # Check caption files
            caption_path = os.path.join(dataset_folder, f"image_{i}.txt")
            assert os.path.exists(caption_path)
            
            with open(caption_path, 'r') as f:
                assert f.read() == captions[i]

def test_handler_missing_required_fields(setup_mocks):
    # Test with missing required fields
    job = {
        "id": "test-job-id",
        "input": {
            "images": ["https://example.com/image1.jpg"],
            # Missing "name" field
            "gender": "M"
        }
    }
    
    # Call the handler
    result = rp_handler.handler(job)
    
    # Verify the error
    assert "error" in result
    assert "Missing required field: name" in result["error"]

def test_handler_empty_images(setup_mocks):
    # Test with empty images list
    job = {
        "id": "test-job-id",
        "input": {
            "images": [],  # Empty list
            "name": "test-model",
            "gender": "F"
        }
    }
    
    # Call the handler
    result = rp_handler.handler(job)
    
    # Verify the error
    assert "error" in result
    assert "images must be a non-empty list" in result["error"]

def test_handler_invalid_image_url(setup_mocks):
    # Test with invalid image URL
    job = {
        "id": "test-job-id",
        "input": {
            "images": ["not-a-url"],  # Invalid URL
            "name": "test-model",
            "gender": "F"
        }
    }
    
    # Call the handler
    result = rp_handler.handler(job)
    
    # Verify the error
    assert "error" in result
    assert "Invalid image URL" in result["error"]

def test_create_captioned_dataset(setup_mocks, mock_requests):
    # Test the create_captioned_dataset function directly
    image_urls = ["https://example.com/image1.jpg", "https://example.com/image2.jpg"]
    
    # Mock run_captioning to return predictable captions
    expected_captions = ["Caption for image 0", "Caption for image 1"]
    with patch('rp_handler.run_captioning', return_value=expected_captions):
        # Call the function
        dataset_folder, captions = rp_handler.create_captioned_dataset(image_urls, False)
        
        # Verify the dataset was created
        assert os.path.exists(dataset_folder)
        
        # Verify image files were created
        assert os.path.exists(os.path.join(dataset_folder, "image_0.jpg"))
        assert os.path.exists(os.path.join(dataset_folder, "image_1.jpg"))
        
        # Verify caption files were created
        assert os.path.exists(os.path.join(dataset_folder, "image_0.txt"))
        assert os.path.exists(os.path.join(dataset_folder, "image_1.txt"))
        
        # Verify the returned captions match expected
        assert captions == expected_captions
        
        # Verify the content of the caption files
        with open(os.path.join(dataset_folder, "image_0.txt"), "r") as f:
            caption_0 = f.read()
        with open(os.path.join(dataset_folder, "image_1.txt"), "r") as f:
            caption_1 = f.read()
        
        assert caption_0 == "Caption for image 0"
        assert caption_1 == "Caption for image 1"

def test_get_config(setup_mocks):
    # Test the get_config function
    name = "test-model"
    dataset_dir = "/path/to/dataset"
    output_dir = "/path/to/output"
    custom_steps = 2000
    
    # Call the function
    config = rp_handler.get_config(name, dataset_dir, output_dir, steps=custom_steps)
    
    # Verify the config structure
    assert isinstance(config, dict) or hasattr(config, 'items')
    
    # Convert OrderedDict to dict for easier assertion
    if hasattr(config, 'items'):
        config_dict = {k: v for k, v in config.items()}
    else:
        config_dict = config
    
    # Check specific paths and values
    assert config_dict["config"]["name"] == name
    process = config_dict["config"]["process"][0]
    assert process["training_folder"] == output_dir
    assert process["datasets"][0]["folder_path"] == dataset_dir
    
    # Check that custom steps value is used
    assert process["train"]["steps"] == custom_steps
    
    # Use the helper function to verify the rest of the structure
    verify_config_structure(config_dict)

def test_handler_exception_handling(setup_mocks):
    # Test exception handling in the handler
    job = {
        "id": "test-job-id",
        "input": {
            "images": ["https://example.com/image1.jpg"],
            "name": "test-model",
            "gender": "F"
        }
    }
    
    # Force an exception in create_captioned_dataset
    with patch('rp_handler.create_captioned_dataset', side_effect=Exception("Test exception")):
        # Call the handler
        result = rp_handler.handler(job)
        
        # Verify the error
        assert "error" in result
        assert "Test exception" in result["error"]

def test_subprocess_failure(setup_mocks, mock_requests):
    # Test handling of subprocess failure
    job = {
        "id": "test-job-id",
        "input": {
            "images": ["https://example.com/image1.jpg"],
            "name": "test-model",
            "gender": "F"
        }
    }
    
    # Mock subprocess.Popen to return a failure
    class MockFailureProcess:
        def __init__(self):
            self.output_lines = ["Error output line"]
            self.current_line = 0
            self.return_code = 1
            
        def readline(self):
            if self.current_line < len(self.output_lines):
                line = self.output_lines[self.current_line] + "\n"
                self.current_line += 1
                return line
            return ""
            
        def poll(self):
            if self.current_line < len(self.output_lines):
                return None
            return self.return_code
    
    mock_process = MockFailureProcess()
    mock_process.stdout = mock_process
    with patch('rp_handler.subprocess.Popen', return_value=mock_process):
        with patch('rp_handler.run_captioning', return_value=["Test caption"]):
            # Call the handler
            result = rp_handler.handler(job)
            
            # Verify the error
            assert "error" in result
            assert "Training failed with exit code 1" in result["error"]

@patch('rp_handler.logger')
def test_run_captioning_successful_logging(mock_logger, mock_cuda, mock_image_generation, temp_dir):
    """Test that run_captioning logs correctly during successful execution"""
    # Setup test data
    image_urls = ["https://example.com/image1.jpg", "https://example.com/image2.jpg"]
    
    # Mock the model and processor
    with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model_cls, \
         patch('transformers.AutoProcessor.from_pretrained') as mock_processor_cls, \
         patch('requests.get') as mock_get:
        
        # Setup mock responses
        mock_get.return_value = MockResponse()
        
        # Setup model and processor behavior
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_model_cls.return_value = mock_model
        mock_processor_cls.return_value = mock_processor
        
        # Configure processor to return specific captions
        mock_processor.post_process_generation.side_effect = [
            {"<DETAILED_CAPTION>": "The image shows a person walking"},
            {"<DETAILED_CAPTION>": "The image shows a sunset"}
        ]
        
        # Run the captioning
        result_captions = rp_handler.run_captioning(image_urls, False)
        
        # Verify results
        assert len(result_captions) == 2
        assert result_captions[0] == "a person walking"
        assert result_captions[1] == "a sunset"
        
        # Verify logging calls
        mock_logger.info.assert_any_call("Starting image captioning process", 
                                        extra={"num_images": 2})
        mock_logger.info.assert_any_call("Loading Florence-2 model and processor", 
                                        extra={"model_name": 'microsoft/Florence-2-large'})
        mock_logger.info.assert_any_call("Captioning process completed", 
                                        extra={"num_images": 2})
        
        # Verify that processing logs were called for each image
        processing_calls = [call for call in mock_logger.info.call_args_list 
                           if len(call[0]) > 0 and "Processing image" in call[0][0]]
        assert len(processing_calls) == 2
        
        # Verify that caption generation success logs were called
        success_calls = [call for call in mock_logger.info.call_args_list 
                        if len(call[0]) > 0 and "Caption generated successfully" in call[0][0]]
        assert len(success_calls) == 2

@patch('rp_handler.logger')
def test_run_captioning_with_image_loading_error_logging(mock_logger, mock_cuda, mock_image_generation, temp_dir):
    """Test that run_captioning logs errors correctly when image loading fails"""
    # Setup test data with one valid and one invalid URL
    image_urls = ["https://example.com/valid.jpg", "https://example.com/invalid.jpg"]
    
    # Mock the model and processor
    with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model_cls, \
         patch('transformers.AutoProcessor.from_pretrained') as mock_processor_cls, \
         patch('requests.get') as mock_get:
        
        # Setup model and processor behavior
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_model_cls.return_value = mock_model
        mock_processor_cls.return_value = mock_processor
        
        # Configure processor to return a caption for successful images
        mock_processor.post_process_generation.return_value = {
            "<DETAILED_CAPTION>": "The image shows a person walking"
        }
        
        # Mock requests.get to succeed for first URL and fail for second
        def mock_get_side_effect(url):
            if url == "https://example.com/valid.jpg":
                return MockResponse()  # Successful response
            else:
                # Simulate HTTP error for invalid URL by raising an exception
                raise Exception("HTTP Error: 404")
        
        mock_get.side_effect = mock_get_side_effect
        
        # Run the captioning
        result_captions = rp_handler.run_captioning(image_urls, False)
        
        # Verify results
        assert len(result_captions) == 2
        assert result_captions[0] == "a person walking"  # First image should have a caption
        assert result_captions[1] == ""  # Second image should have empty string due to loading failure
        
        # Verify error logging for the failed image
        error_calls = [call for call in mock_logger.error.call_args_list 
                      if len(call[0]) > 0 and "Error loading image" in call[0][0]]
        assert len(error_calls) == 1
        
        # Verify the error call contains the expected information
        error_call = error_calls[0]
        assert error_call[1]['extra']['image_url'] == "https://example.com/invalid.jpg"
        assert "HTTP Error: 404" in error_call[1]['extra']['error']

@patch('rp_handler.logger')
def test_run_captioning_model_loading_error_logging(mock_logger, mock_cuda, mock_image_generation, temp_dir):
    """Test that run_captioning logs errors correctly when model loading fails"""
    # Setup test data
    image_urls = ["https://example.com/image1.jpg"]
    
    # Mock the model loading to fail
    with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model_cls:
        
        # Make model loading fail
        mock_model_cls.side_effect = Exception("Failed to load model")
        
        # Run the captioning and expect exception
        with pytest.raises(Exception) as excinfo:
            rp_handler.run_captioning(image_urls, False)
        
        # Verify error message
        assert "Failed to load model" in str(excinfo.value)
        
        # Verify error logging
        error_calls = [call for call in mock_logger.error.call_args_list 
                      if len(call[0]) > 0 and "Failed to load model or processor" in call[0][0]]
        assert len(error_calls) == 1
        
        # Verify the error call contains the expected information
        error_call = error_calls[0]
        assert error_call[1]['extra']['model_name'] == 'microsoft/Florence-2-large'
        assert "Failed to load model" in error_call[1]['extra']['error']

def test_handler_caption_logging(setup_mocks, mock_requests):
    """Test that generated captions are properly logged"""
    # Test with valid input
    job = {
        "id": "test-job-id",
        "input": {
            "images": ["https://example.com/image1.jpg", "https://example.com/image2.jpg"],
            "name": "test-model",
            "gender": "F"
        }
    }
    
    # Mock the run_captioning function to return predictable captions
    expected_captions = ["Expected caption for image 0", "Expected caption for image 1"]
    
    with patch('rp_handler.run_captioning', return_value=expected_captions):
        with patch('rp_handler.logger') as mock_logger:
            # Call the handler
            result = rp_handler.handler(job)
            
            # Find the "Saved config" log call
            saved_config_calls = [
                call for call in mock_logger.info.call_args_list
                if call[0][0] == "Saved config"
            ]
            
            # Verify that there's exactly one "Saved config" call
            assert len(saved_config_calls) == 1
            
            # Get the call and verify its contents
            call_args, call_kwargs = saved_config_calls[0]
            assert call_args[0] == "Saved config"
            
            extra = call_kwargs.get('extra', {})
            assert extra.get('job_name') == "test-model"
            assert extra.get('captions') == expected_captions
            assert 'config_path' in extra
            assert 'dataset_folder_path' in extra
            assert 'dataset_folder_relative_path' in extra

def test_run_captioning_image_loading_failure(mock_cuda, mock_image_generation, temp_dir):
    """Test that run_captioning sets empty string captions when image loading fails"""
    # Setup test data with one valid and one invalid URL
    image_urls = ["https://example.com/valid.jpg", "https://example.com/invalid.jpg"]
    
    # Mock the model and processor
    with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model_cls, \
         patch('transformers.AutoProcessor.from_pretrained') as mock_processor_cls, \
         patch('requests.get') as mock_get:
        
        # Setup model and processor behavior
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_model_cls.return_value = mock_model
        mock_processor_cls.return_value = mock_processor
        
        # Configure processor to return a caption for successful images
        mock_processor.post_process_generation.return_value = {
            "<DETAILED_CAPTION>": "The image shows a person walking"
        }
        
        # Mock requests.get to succeed for first URL and fail for second
        def mock_get_side_effect(url):
            if url == "https://example.com/valid.jpg":
                return MockResponse()  # Successful response
            else:
                # Simulate HTTP error for invalid URL by raising an exception
                raise Exception("HTTP Error: 404")
        
        mock_get.side_effect = mock_get_side_effect
        
        # Run the captioning
        result_captions = rp_handler.run_captioning(image_urls, False)
        
        # Verify results
        assert len(result_captions) == 2
        assert result_captions[0] == "a person walking"  # First image should have a caption
        assert result_captions[1] == ""  # Second image should have empty string due to loading failure
        
        # Verify that requests.get was called for both URLs
        assert mock_get.call_count == 2
        
        # Verify that the processor was only called once (for the successful image)
        assert mock_processor.post_process_generation.call_count == 1
