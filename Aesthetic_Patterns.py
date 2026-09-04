"""
Photo Pattern Generator for TikTok (Improved)
"""

import math
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

MODE = "aesthetic"
SOURCE_FOLDERS = [
    r"D:\photos\fancy\estetica",
    r"D:\photos\larp\autos",
    r"D:\photos\larp\life",
]

OUTPUT_FOLDER = r"D:\photos\prueba 2"      
NUM_PATTERNS = 2                
PHOTOS_PER_PATTERN = 6          
NUM_AESTHETIC_CLUSTERS = 2       
MAX_FRACTION_PER_FOLDER = 2 / 3  
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
ANALYSIS_THUMBNAIL_SIZE = (100, 100) 


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


def extract_pool(source_folders: list[str]) -> tuple[list[Path], dict[Path, str]]:
    pool: list[Path] = []
    folder_of: dict[Path, str] = {}

    for folder in source_folders:
        available_photos = get_photos(folder)
        max_allowed = max(1, int(len(available_photos) * MAX_FRACTION_PER_FOLDER))
        quantity = random.randint(1, max_allowed)
        selected = random.sample(available_photos, quantity)

        pool.extend(selected)
        for photo in selected:
            folder_of[photo] = folder
        print(f"  → {quantity}/{len(available_photos)} photos taken from '{folder}' (max allowed: {max_allowed})")

    return pool, folder_of


def analyze_photo(photo_path: Path) -> np.ndarray:
    with Image.open(photo_path) as img:
        img = img.convert("RGB")
        img = img.resize(ANALYSIS_THUMBNAIL_SIZE)
        pixels = np.asarray(img, dtype=np.float32) / 255.0

    avg_r = pixels[:, :, 0].mean()
    avg_g = pixels[:, :, 1].mean()
    avg_b = pixels[:, :, 2].mean()

    brightness = 0.299 * avg_r + 0.587 * avg_g + 0.114 * avg_b
    max_c = pixels.max(axis=2)
    min_c = pixels.min(axis=2)
    saturation = (max_c - min_c).mean()
    warmth = avg_r - avg_b

    return np.array([avg_r, avg_g, avg_b, brightness, saturation, warmth])


def cluster_by_aesthetic(photos: list[Path], num_clusters: int):
    print(f"\nAnalyzing the visual aesthetic of {len(photos)} photos...")

    features = []
    valid_photos = []
    for photo in photos:
        try:
            features.append(analyze_photo(photo))
            valid_photos.append(photo)
        except Exception as e:
            print(f"  ⚠ Skipping '{photo.name}': could not analyze it ({e})")

    if len(valid_photos) < num_clusters:
        raise ValueError("Not enough photos to analyze. Reduce NUM_AESTHETIC_CLUSTERS.")

    feature_matrix = np.array(features)
    
    # Scale features so no single metric dominates Euclidean distance
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(feature_matrix)
    feature_by_photo = {photo: feat for photo, feat in zip(valid_photos, scaled_matrix)}

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_matrix)

    clusters: dict[int, list[Path]] = {i: [] for i in range(num_clusters)}
    for photo, label in zip(valid_photos, labels):
        clusters[label].append(photo)

    raw_centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    
    print("\nDetected aesthetic groups:")
    for cluster_id, cluster_photos in clusters.items():
        avg_r, avg_g, avg_b, brightness, saturation, warmth = raw_centroids[cluster_id]
        description = describe_aesthetic(brightness, saturation, warmth)
        print(f"  Cluster {cluster_id}: {len(cluster_photos)} photos — {description}")

    return clusters, kmeans, feature_by_photo, valid_photos


def balanced_allocate(counts: dict[str, int], n: int) -> dict[str, int]:
    """
    Splits n slots across folders as evenly as possible, respecting each
    folder's available count. Works round-robin: each eligible folder gets
    +1 per pass (in randomized order), so any leftover from integer
    division is spread out instead of dumped on whichever folder happened
    to be sorted first. Returns fewer than n total only if every folder
    runs out of supply first.
    """
    allocation = {folder: 0 for folder in counts}
    folders = list(counts.keys())
    total_allocated = 0

    while total_allocated < n:
        made_progress = False
        random.shuffle(folders)
        for folder in folders:
            if total_allocated >= n:
                break
            if allocation[folder] < counts[folder]:
                allocation[folder] += 1
                total_allocated += 1
                made_progress = True
        if not made_progress:
            break  # every folder is fully used up

    return allocation


def widen_cluster(
    cluster_photos: list[Path],
    cluster_id: int,
    kmeans: KMeans,
    feature_by_photo: dict,
    all_photos: list[Path],
    target_size: int,
    folder_of: dict[Path, str] | None = None,
) -> list[Path]:
    """
    If a cluster has fewer photos than target_size, widen it by pulling in
    the closest-matching photos from the rest of the pool. Those photos
    aren't removed from their original cluster — they're simply also
    considered a reasonable match for this one.

    The extra photos are borrowed in a balanced way: rather than just
    grabbing whichever photos are closest overall (which could pull, say,
    7 from one folder and 2 from another), the shortfall is split as
    evenly as possible across the folders that have a matching photo
    available. Within each folder, the closest aesthetic match is still
    picked first.
    """
    if len(cluster_photos) >= target_size:
        return cluster_photos

    centroid = kmeans.cluster_centers_[cluster_id]
    already_in = set(cluster_photos)
    candidates = [p for p in all_photos if p not in already_in]
    candidates_sorted = sorted(
        candidates,
        key=lambda p: np.linalg.norm(feature_by_photo[p] - centroid)
    )

    needed = target_size - len(cluster_photos)
    borrowed: list[Path] = []

    if folder_of is not None:
        candidates_by_folder: dict[str, list[Path]] = defaultdict(list)
        for p in candidates_sorted:
            candidates_by_folder[folder_of.get(p, "unknown")].append(p)

        available_counts = {folder: len(photos) for folder, photos in candidates_by_folder.items()}
        allocation = balanced_allocate(available_counts, needed)

        for folder, count in allocation.items():
            borrowed.extend(candidates_by_folder[folder][:count])
    else:
        borrowed = candidates_sorted[:needed]

    if borrowed:
        print(f"  ↳ Widening group: added {len(borrowed)} matching photos (balanced across folders).")

    return cluster_photos + borrowed


def describe_aesthetic(brightness: float, saturation: float, warmth: float) -> str:
    light_desc = "bright" if brightness > 0.55 else ("dark" if brightness < 0.35 else "mid-toned")
    sat_desc = "vivid" if saturation > 0.25 else ("muted" if saturation < 0.12 else "balanced")
    warm_desc = "warm" if warmth > 0.05 else ("cool" if warmth < -0.05 else "neutral")
    return f"{light_desc}, {sat_desc}, {warm_desc} tones"


def sample_diverse_pattern(
    photos: list[Path], folder_of: dict[Path, str], n: int, max_per_folder: int
) -> list[Path]:
    by_folder: dict[str, list[Path]] = defaultdict(list)
    for photo in photos:
        by_folder[folder_of.get(photo, "unknown")].append(photo)

    for bucket in by_folder.values():
        random.shuffle(bucket)

    folders = list(by_folder.keys())
    random.shuffle(folders)

    selected: list[Path] = []
    counts: dict[str, int] = defaultdict(int)
    positions: dict[str, int] = {folder: 0 for folder in folders}

    while len(selected) < n:
        made_progress = False
        random.shuffle(folders)
        for folder in folders:
            if len(selected) >= n:
                break
            bucket = by_folder[folder]
            pos = positions[folder]
            if counts[folder] < max_per_folder and pos < len(bucket):
                selected.append(bucket[pos])
                positions[folder] += 1
                counts[folder] += 1
                made_progress = True
        if not made_progress:
            break

    if len(selected) < n:
        already_selected = set(selected)
        remaining = [p for p in photos if p not in already_selected]
        remaining.sort(key=lambda p: counts[folder_of.get(p, "unknown")])
        for p in remaining:
            if len(selected) >= n:
                break
            selected.append(p)
            counts[folder_of.get(p, "unknown")] += 1

    random.shuffle(selected)
    return selected


def generate_patterns(
    photos: list[Path],
    num_patterns: int,
    photos_per_pattern: int,
    folder_of: dict[Path, str] | None = None,
    max_fraction_per_folder: float | None = None,
) -> list[tuple[Path, ...]]:
    if photos_per_pattern > len(photos):
        raise ValueError("Not enough photos to fulfill pattern size.")

    use_diversity_cap = folder_of is not None and max_fraction_per_folder is not None
    max_per_folder = None
    if use_diversity_cap:
        max_per_folder = max(1, math.floor(max_fraction_per_folder * photos_per_pattern))

    generated_patterns: set[tuple[Path, ...]] = set()
    attempts = 0
    max_attempts = num_patterns * 200 

    while len(generated_patterns) < num_patterns and attempts < max_attempts:
        if use_diversity_cap:
            selection = sample_diverse_pattern(photos, folder_of, photos_per_pattern, max_per_folder)
        else:
            selection = random.sample(photos, photos_per_pattern) 

        pattern = tuple(selection)
        generated_patterns.add(pattern)
        attempts += 1

    if len(generated_patterns) < num_patterns:
        print(
            f"  ⚠ Only {len(generated_patterns)} unique patterns could be generated "
            f"(asked for {num_patterns}). With {len(photos)} photos and {photos_per_pattern} "
            f"per pattern, the possible combinations are limited."
        )

    return list(generated_patterns)


def save_patterns(patterns: list[tuple[Path, ...]], output_folder: str):
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True, parents=True)

    for i, pattern in enumerate(patterns, start=1):
        pattern_folder = output_path / f"pattern_{i:02d}"
        pattern_folder.mkdir(exist_ok=True)

        for order, photo in enumerate(pattern, start=1):
            new_name = f"{order:02d}_{photo.name}"
            target_path = pattern_folder / new_name
            
            # Fast, space-saving hardlink with an immediate copy fallback
            try:
                target_path.hardlink_to(photo)
            except OSError:
                shutil.copy2(photo, target_path)

        print(f"  ✔ {pattern_folder.name} created with {len(pattern)} photos")


def run_random_mode():
    pool, folder_of = extract_pool(SOURCE_FOLDERS)
    patterns = generate_patterns(
        pool, NUM_PATTERNS, PHOTOS_PER_PATTERN,
        folder_of=folder_of, max_fraction_per_folder=MAX_FRACTION_PER_FOLDER,
    )
    save_patterns(patterns, OUTPUT_FOLDER)


def run_aesthetic_mode():
    pool, folder_of = extract_pool(SOURCE_FOLDERS)
    clusters, kmeans, feature_by_photo, valid_photos = cluster_by_aesthetic(pool, NUM_AESTHETIC_CLUSTERS)

    output_path = Path(OUTPUT_FOLDER)
    output_path.mkdir(exist_ok=True)

    for cluster_id, cluster_photos in clusters.items():
        if len(cluster_photos) == 0:
            continue

        cluster_name = f"aesthetic_{cluster_id + 1:02d}"
        working_photos = widen_cluster(
            cluster_photos, cluster_id, kmeans, feature_by_photo, valid_photos, PHOTOS_PER_PATTERN,
            folder_of=folder_of,
        )

        patterns = generate_patterns(
            working_photos, NUM_PATTERNS, PHOTOS_PER_PATTERN,
            folder_of=folder_of, max_fraction_per_folder=MAX_FRACTION_PER_FOLDER,
        )
        save_patterns(patterns, str(output_path / cluster_name))


def main():
    if MODE == "random":
        run_random_mode()
    elif MODE == "aesthetic":
        run_aesthetic_mode()
    else:
        raise ValueError(f"Invalid MODE: '{MODE}'. Use 'random' or 'aesthetic'.")

if __name__ == "__main__":
    main()