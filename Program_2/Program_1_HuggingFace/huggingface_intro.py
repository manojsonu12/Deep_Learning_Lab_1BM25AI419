from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from pathlib import Path

# Create output folder
BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "Results" / "HuggingFace"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("HUGGING FACE HUB INTRODUCTION")
print("-" * 40)

# 1. Connect to Hugging Face Hub
api = HfApi()

model_id = "distilbert/distilbert-base-uncased"

try:
    model = api.model_info(model_id)

    print("\nModel Name:", model.id)
    print("Model Author:", model.author)
    print("Model Library:", model.library_name)

    # 2. Download model card
    card_path = hf_hub_download(
        repo_id=model_id,
        filename="README.md"
    )

    with open(card_path, "r", encoding="utf-8") as file:
        card_content = file.read()

    with open(OUTPUT_DIR / "model_card.txt", "w",
              encoding="utf-8") as file:
        file.write(card_content)

    print("\nModel card downloaded successfully!")

except Exception as e:
    print("Model information error:", e)

# 3. Load dataset
try:
    print("\nLoading Rotten Tomatoes dataset...")

    dataset = load_dataset(
        "cornell-movie-review-data/rotten_tomatoes"
    )

    print("\nDataset Information:")
    print(dataset)

    # Display one sample
    sample = dataset["train"][0]

    print("\nSample Data:")
    print("Text:", sample["text"])
    print("Label:", sample["label"])

    # Save sample
    with open(OUTPUT_DIR / "dataset_sample.txt",
              "w", encoding="utf-8") as file:
        file.write(str(sample))

    print("\nDataset loaded successfully!")

except Exception as e:
    print("Dataset loading error:", e)

print("\nProgram completed!")