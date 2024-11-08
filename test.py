from flux_train_ui import start_training

start_training(
    lora_name = "test-lora",
    concept_sentence = False,
    steps = 1000,
    lr = 0.0004,
    rank = 16,
    model_to_train = "dev",
    low_vram = False,
    dataset_folder = "datasets/test",
    sample_1 = False,
    sample_2 = False,
    sample_3 = False,
    use_more_advanced_options = False,
    more_advanced_options = "device: cuda:0"
)