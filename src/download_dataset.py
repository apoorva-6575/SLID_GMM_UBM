from datasets import load_dataset

# ---------------------------------------------------------
# Languages to download (Development Phase)
# ---------------------------------------------------------
languages = [
    "hindi",
    "bengali",
    "tamil"
]

# ---------------------------------------------------------
# Download each language
# ---------------------------------------------------------
for language in languages:

    print(f"\nDownloading {language}...\n")

    dataset = load_dataset(
        "ai4bharat/Shrutilipi",
        language,
        cache_dir="dataset"
    )

    print(f"{language} downloaded successfully!")