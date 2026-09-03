import random
import shutil
from pathlib import Path

SOURCE_FOLDERS = {
    "photos_folder1": 4,
    "photos_folder2": 3,
    "photos_folder3": 5,
    "photos_folder4": 2,
    "photos_folder5": 4,
}

"""⬇️⬇️Edit the variables in this section⬇️⬇️"""

OUTPUT_FOLDER = "patterns"
NUM_PATTERNS = 10
PHOTOS_PER_PATTERN = 5
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}

def get_photos(folder: str) -> list[Path]:
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder '{folder}' does not exist")

    photos = [
        f for f in folder_path.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
    ]
    if not photos:
        raise ValueError(f"No valid photos were found in '{folder}'")
    return photos


def extract_pool(source_folders: dict[str, int]) -> list[Path]:
    if len(source_folders) < 5:
        print(f"⚠ You configured {len(source_folders)} folders (a minimum of 5 is recommended).")

    pool: list[Path] = []
    for folder, quantity in source_folders.items():
        available_photos = get_photos(folder)

        if quantity > len(available_photos):
            raise ValueError(
                f"You asked for {quantity} photos from '{folder}' but only {len(available_photos)} are available."
            )

        selected = random.sample(available_photos, quantity)  # no repeats within the folder
        pool.extend(selected)
        print(f"  → {quantity} photos extracted from '{folder}'")

    return pool


def generate_patterns(photos: list[Path], num_patterns: int, photos_per_pattern: int) -> list[tuple[Path, ...]]:
    if photos_per_pattern > len(photos):
        raise ValueError(
            f"You asked for {photos_per_pattern} photos per pattern but the total pool only has {len(photos)} photos."
        )

    generated_patterns: set[tuple[Path, ...]] = set()
    attempts = 0
    max_attempts = num_patterns * 200  # safety margin

    while len(generated_patterns) < num_patterns and attempts < max_attempts:
        selection = random.sample(photos, photos_per_pattern)  # no repeats within the pattern
        pattern = tuple(selection)
        generated_patterns.add(pattern)
        attempts += 1

    if len(generated_patterns) < num_patterns:
        print(
            f"⚠ Only {len(generated_patterns)} unique patterns could be generated "
            f"(you asked for {num_patterns}). With {len(photos)} photos in the pool and {photos_per_pattern} "
            f"per pattern, the possible combinations are limited."
        )

    return list(generated_patterns)


def save_patterns(patterns: list[tuple[Path, ...]], output_folder: str):
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    for i, pattern in enumerate(patterns, start=1):
        pattern_folder = output_path / f"pattern_{i:02d}"
        pattern_folder.mkdir(exist_ok=True)

        for order, photo in enumerate(pattern, start=1):
            new_name = f"{order:02d}_{photo.name}"
            shutil.copy2(photo, pattern_folder / new_name)

        print(f"✔ {pattern_folder.name} created with {len(pattern)} photos")


def main():
    print("Extracting photos from the source folders...")
    pool = extract_pool(SOURCE_FOLDERS)
    print(f"\nTotal pool: {len(pool)} combined photos")

    print(f"\nGenerating {NUM_PATTERNS} patterns of {PHOTOS_PER_PATTERN} photos each...")
    patterns = generate_patterns(pool, NUM_PATTERNS, PHOTOS_PER_PATTERN)

    print(f"\nSaving patterns to '{OUTPUT_FOLDER}'...")
    save_patterns(patterns, OUTPUT_FOLDER)

    print(f"\n✅ Done. {len(patterns)} patterns were generated in the '{OUTPUT_FOLDER}' folder.")


if __name__ == "__main__":
    main()