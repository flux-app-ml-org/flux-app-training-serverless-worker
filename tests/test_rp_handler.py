# tests/test_rp_handler.py
import sys
import pytest
from unittest.mock import patch, MagicMock, mock_open, ANY
import json
import os
import shutil
import tempfile
from io import BytesIO
from PIL import Image as PILImage  # We'll use the real PIL for image operations

# Mock heavy dependencies before importing the module under test
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['runpod'] = MagicMock()
sys.modules['loki_logger_handler'] = MagicMock()
sys.modules['loki_logger_handler.loki_logger_handler'] = MagicMock()
sys.modules['toolkit'] = MagicMock()
sys.modules['toolkit.job'] = MagicMock()

# Now we can safely import the module
import rp_handler

# Create mock classes for complex objects
class MockTensor:
    def __init__(self, *args, **kwargs):
        pass
    
    def to(self, *args, **kwargs):
        return self

class MockProcessorOutput:
    def __init__(self):
        self.data = {"input_ids": MockTensor(), "pixel_values": MockTensor()}
    
    def __getitem__(self, key):
        return self.data[key]
    
    def to(self, *args, **kwargs):
        return self

class MockModel:
    def __init__(self, *args, **kwargs):
        pass
    
    def to(self, device):
        return self
    
    def generate(self, **kwargs):
        return [1, 2, 3]  # Dummy token IDs

class MockProcessor:
    def __init__(self, *args, **kwargs):
        pass
    
    def __call__(self, **kwargs):
        return MockProcessorOutput()
    
    def batch_decode(self, ids, **kwargs):
        return ["Generated text"]
    
    def post_process_generation(self, text, **kwargs):
        return {"<DETAILED_CAPTION>": "The image shows a person"}

class MockJob:
    def __init__(self, *args, **kwargs):
        self.config = args[0]
        self.name = args[1]
        self.run_called = False
        self.cleanup_called = False
    
    def run(self):
        self.run_called = True
    
    def cleanup(self):
        self.cleanup_called = True

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

@pytest.fixture
def setup_mocks(temp_dir):
    # Set up all the necessary mocks
    rp_handler.torch.cuda.is_available = MagicMock(return_value=True)
    rp_handler.torch.float16 = "float16"
    
    # Mock model and processor
    mock_model = MockModel()
    rp_handler.AutoModelForCausalLM.from_pretrained = MagicMock(return_value=mock_model)
    rp_handler.AutoProcessor.from_pretrained = MagicMock(return_value=MockProcessor())
    
    # Keep real PIL.Image for actual image operations
    original_pil_open = PILImage.open
    
    # Create a mock job that we can inspect
    mock_job = MockJob(None, None)
    rp_handler.get_job = MagicMock(return_value=mock_job)
    
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
            
            # Create metadata entry
            jsonl_file_path = os.path.join(dataset_dir, "metadata.jsonl")
            with open(jsonl_file_path, "a") as jsonl_file:
                data = {"file_name": f"image_{index}.jpg", "prompt": captions[index]}
                jsonl_file.write(json.dumps(data) + "\n")
        
        return dataset_dir
    
    rp_handler.create_dataset = mock_create_dataset
    
    # Mock run_captioning to return predictable captions
    def mock_run_captioning(images, concept_sentence, *captions):
        return [f"Caption for image {i}" for i in range(len(images))]
    
    rp_handler.run_captioning = mock_run_captioning
    
    yield {
        "temp_dir": temp_dir,
        "mock_job": mock_job
    }
    
    # Restore original functions
    rp_handler.create_dataset = original_create_dataset
    PILImage.open = original_pil_open

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
        
        # Verify metadata file was created and contains expected content
        metadata_path = os.path.join(dataset_path, "metadata.jsonl")
        assert os.path.exists(metadata_path)
        
        with open(metadata_path, "r") as f:
            metadata_lines = f.readlines()
            assert len(metadata_lines) == len(expected_captions)
            
            for i, line in enumerate(metadata_lines):
                entry = json.loads(line)
                assert entry["prompt"] == expected_captions[i]
                assert entry["file_name"] == f"image_{i}.jpg"
        
        # Verify job was run
        mock_job = setup_mocks["mock_job"]
        assert mock_job.run_called
        assert mock_job.cleanup_called
        
        # Verify get_job was called with correct parameters
        rp_handler.get_job.assert_called_once()
        config_arg = rp_handler.get_job.call_args[0][0]
        name_arg = rp_handler.get_job.call_args[0][1]
        assert name_arg == "test-model"
        assert isinstance(config_arg, dict) or hasattr(config_arg, 'items')

def test_create_captioned_dataset_content(setup_mocks, mock_requests):
    """Test that the captions are correctly written to text files"""
    image_urls = ["https://example.com/image1.jpg", "https://example.com/image2.jpg"]
    
    # Create specific test captions
    test_captions = ["Test caption for image 0", "Test caption for image 1"]
    
    # Mock the run_captioning function to return our test captions
    with patch('rp_handler.run_captioning', return_value=test_captions):
        # Call the function
        dataset_folder = rp_handler.create_captioned_dataset(image_urls, False)
        
        # Verify the dataset was created
        assert os.path.exists(dataset_folder)
        
        # Verify the content of the caption files matches our test captions
        for i, caption in enumerate(test_captions):
            caption_file_path = os.path.join(dataset_folder, f"image_{i}.txt")
            assert os.path.exists(caption_file_path)
            
            with open(caption_file_path, "r") as f:
                file_content = f.read()
                assert file_content == caption
        
        # Verify metadata file contains our captions
        metadata_path = os.path.join(dataset_folder, "metadata.jsonl")
        assert os.path.exists(metadata_path)
        
        with open(metadata_path, "r") as f:
            metadata_lines = f.readlines()
            assert len(metadata_lines) == len(test_captions)
            
            for i, line in enumerate(metadata_lines):
                entry = json.loads(line)
                assert entry["prompt"] == test_captions[i]
                assert entry["file_name"] == f"image_{i}.jpg"

def test_generate_caption_for_image(setup_mocks):
    """Test the single image captioning function"""
    # Create a test image
    test_image = PILImage.new('RGB', (100, 100), color='red')
    
    # Create mock processor and model
    mock_processor = MockProcessor()
    mock_model = MockModel()
    
    # Set up the expected return value for post_process_generation
    expected_caption = "a beautiful sunset over mountains"
    mock_processor.post_process_generation = MagicMock(
        return_value={"<DETAILED_CAPTION>": f"The image shows {expected_caption}"}
    )
    
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

def test_handler_invalid_gender(setup_mocks):
    # Test with invalid gender
    job = {
        "id": "test-job-id",
        "input": {
            "images": ["https://example.com/image1.jpg"],
            "name": "test-model",
            "gender": "X"  # Invalid gender
        }
    }
    
    # Call the handler
    result = rp_handler.handler(job)
    
    # Verify the error
    assert "error" in result
    assert "gender must be either 'F' or 'M'" in result["error"]

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
    
    # Call the function
    dataset_folder = rp_handler.create_captioned_dataset(image_urls, False)
    
    # Verify the dataset was created
    assert os.path.exists(dataset_folder)
    
    # Verify image files were created
    assert os.path.exists(os.path.join(dataset_folder, "image_0.jpg"))
    assert os.path.exists(os.path.join(dataset_folder, "image_1.jpg"))
    
    # Verify caption files were created
    assert os.path.exists(os.path.join(dataset_folder, "image_0.txt"))
    assert os.path.exists(os.path.join(dataset_folder, "image_1.txt"))
    
    # Verify metadata file was created
    assert os.path.exists(os.path.join(dataset_folder, "metadata.jsonl"))
    
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
    gender = "F"
    
    # Call the function
    config = rp_handler.get_config(name, dataset_dir, output_dir, gender)
    
    # Verify the config structure
    assert isinstance(config, dict) or hasattr(config, 'items')
    
    # Convert OrderedDict to dict for easier assertion
    if hasattr(config, 'items'):
        config_dict = {k: v for k, v in config.items()}
    else:
        config_dict = config
    
    # Check key parts of the config
    assert "job" in config_dict
    assert "config" in config_dict
    assert "meta" in config_dict
    
    # Check the name is set correctly
    assert config_dict["config"]["name"] == name
    
    # Check dataset directory is set correctly
    process = config_dict["config"]["process"][0]
    dataset_config = process["datasets"][0]
    assert dataset_config["folder_path"] == dataset_dir
    
    # Check output directory is set correctly
    assert process["training_folder"] == output_dir

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
