"""General project utilities."""

from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_dir(path):
    """Create a directory if it does not exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def download_shrutilipi_languages(languages=None, cache_dir=None):
    """Download selected AI4Bharat Shrutilipi language splits."""
    selected_languages = languages or ["hindi"]
    target_cache = cache_dir or PROJECT_ROOT / "dataset"

    datasets = {}
    for language in selected_languages:
        print(f"\nDownloading {language}...\n")
        datasets[language] = load_dataset(
            "ai4bharat/Shrutilipi",
            language,
            cache_dir=str(target_cache),
        )
        print(f"{language} downloaded successfully!")

    print("\nAll selected languages downloaded successfully!")
    return datasets
