"""
Photo Pattern Generator for TikTok (Improved)
"""

import concurrent.futures
import json
import math
import random
import re
import shutil
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Solid/near-solid-color photos legitimately have fewer distinct pixel
# values than DOMINANT_COLOR_CLUSTERS — sklearn warns about it, but the
# result is still correct (KMeans handles duplicate points fine).
warnings.filterwarnings("ignore", category=ConvergenceWarning)

MODE = "aesthetic"
SOURCE_FOLDERS = [
    r"D:\photos\fancy\estetica",
    r"D:\photos\larp\autos",
    r"D:\photos\larp\life",
]

OUTPUT_FOLDER = r"D:\photos\prueba 2"      
NUM_PATTERNS = 2                
PHOTOS_PER_PATTERN = 6          
NUM_AESTHETIC_CLUSTERS = "auto"  # int for a fixed number, or "auto" to let the script decide
MAX_FRACTION_PER_FOLDER = 2 / 3  
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
ANALYSIS_THUMBNAIL_SIZE = (100, 100) 

# Optional format filter — leave both as None to use every photo regardless
# of shape (default behavior, nothing changes if you don't touch these).
#   ORIENTATION_FILTER: None, "horizontal", "vertical", or "square"
#   ASPECT_RATIO_FILTER: None, or a specific ratio as "W:H", e.g. "9:16",
#     "4:5", "3:4", "1:1", "16:9". Only meaningful once ORIENTATION_FILTER
#     narrows things down to one orientation — e.g. set ORIENTATION_FILTER
#     to "vertical" and ASPECT_RATIO_FILTER to "9:16" to keep only
#     vertical photos close to that specific shape.
ORIENTATION_FILTER = None
ASPECT_RATIO_FILTER = None
ASPECT_RATIO_TOLERANCE = 0.05  # allow 5% wiggle room so near-matches still count
COVER_PHOTO_MODE = False  # if True, each pattern's most vivid photo is placed first (position 1)

# Caching: remembers each photo's analyzed features (color/brightness/etc.)
# between runs, keyed by file path + modified time + size — so an unchanged
# photo never gets re-analyzed. Speeds up every run after the first.
ENABLE_CACHE = True
CACHE_FILE = "aesthetic_cache.json"

# Parallel analysis: analyzes this many photos at once instead of one at a
# time. Only affects photos not already in the cache.
ANALYSIS_WORKERS = 8


def get_image_dimensions(photo_path: Path) -> tuple[int, int]:
    with Image.open(photo_path) as img:
        return img.size  # (width, height)


def get_orientation(width: int, height: int) -> str:
    if width > height:
        return "horizontal"
    if height > width:
        return "vertical"
    return "square"


def matches_aspect_ratio(width: int, height: int, ratio_str: str, tolerance: float) -> bool:
    try:
        w_part, h_part = ratio_str.split(":")
        target_ratio = float(w_part) / float(h_part)
    except (ValueError, ZeroDivisionError):
        raise ValueError(f"Invalid ASPECT_RATIO_FILTER format: '{ratio_str}'. Use 'W:H', e.g. '9:16'.")

    actual_ratio = width / height
    return abs(actual_ratio - target_ratio) / target_ratio <= tolerance


def passes_format_filter(photo_path: Path) -> bool:
    if ORIENTATION_FILTER is None and ASPECT_RATIO_FILTER is None:
        return True  # no filter configured — don't even bother opening the file

    try:
        width, height = get_image_dimensions(photo_path)
    except Exception:
        return False  # unreadable image, exclude it rather than crash

    if ORIENTATION_FILTER is not None and get_orientation(width, height) != ORIENTATION_FILTER:
        return False

    if ASPECT_RATIO_FILTER is not None and not matches_aspect_ratio(
        width, height, ASPECT_RATIO_FILTER, ASPECT_RATIO_TOLERANCE
    ):
        return False

    return True


def get_photos(folder: str) -> list[Path]:
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder '{folder}' does not exist")

    all_photos = [
        f for f in folder_path.rglob("*")
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
    ]

    if ORIENTATION_FILTER is None and ASPECT_RATIO_FILTER is None:
        photos = all_photos
    else:
        photos = [p for p in all_photos if passes_format_filter(p)]
        excluded = len(all_photos) - len(photos)
        if excluded > 0:
            print(f"  (filtered out {excluded} photos in '{folder}' not matching the format filter)")

    if not photos:
        raise ValueError(f"No valid photos matching the current filters were found in '{folder}'")
    return photos


def extract_pool(source_folders: list[str], min_total: int | None = None) -> tuple[list[Path], dict[Path, str]]:
    """
    Randomly extracts photos from each folder (1 to 2/3 of what's there).
    If min_total is given and the random draw comes up short of it overall
    (bad luck can make every folder roll a low number), it tops folders up
    — respecting their individual max_allowed cap — until min_total is
    reached, or raises a clear error if that's not physically possible.
    """
    available_by_folder: dict[str, list[Path]] = {}
    max_allowed_by_folder: dict[str, int] = {}
    quantity_by_folder: dict[str, int] = {}

    for folder in source_folders:
        available_photos = get_photos(folder)
        max_allowed = max(1, int(len(available_photos) * MAX_FRACTION_PER_FOLDER))
        available_by_folder[folder] = available_photos
        max_allowed_by_folder[folder] = max_allowed
        quantity_by_folder[folder] = random.randint(1, max_allowed)

    total = sum(quantity_by_folder.values())

    if min_total is not None and total < min_total:
        folders_cycle = list(source_folders)
        while total < min_total:
            made_progress = False
            random.shuffle(folders_cycle)
            for folder in folders_cycle:
                if total >= min_total:
                    break
                if quantity_by_folder[folder] < max_allowed_by_folder[folder]:
                    quantity_by_folder[folder] += 1
                    total += 1
                    made_progress = True
            if not made_progress:
                break  # every folder is already at its max_allowed cap

        if total < min_total:
            max_possible = sum(max_allowed_by_folder.values())
            raise ValueError(
                f"Not enough photos available to fill a pattern of {min_total} photos while "
                f"respecting the {MAX_FRACTION_PER_FOLDER:.0%} per-folder cap "
                f"(MAX_FRACTION_PER_FOLDER). Maximum obtainable this way: {max_possible} photos. "
                f"Add more photos to your source folders, raise MAX_FRACTION_PER_FOLDER, "
                f"or lower PHOTOS_PER_PATTERN."
            )

    pool: list[Path] = []
    folder_of: dict[Path, str] = {}

    for folder in source_folders:
        available_photos = available_by_folder[folder]
        quantity = quantity_by_folder[folder]
        selected = random.sample(available_photos, quantity)

        pool.extend(selected)
        for photo in selected:
            folder_of[photo] = folder
        print(
            f"  → {quantity}/{len(available_photos)} photos taken from '{folder}' "
            f"(max allowed: {max_allowed_by_folder[folder]})"
        )

    return pool, folder_of


DOMINANT_COLOR_CLUSTERS = 3  # how many candidate colors to consider per photo


def get_dominant_color(pixels_flat: np.ndarray, k: int = DOMINANT_COLOR_CLUSTERS) -> np.ndarray:
    """
    Finds the most prevalent color in a photo via a quick per-photo
    k-means, instead of a flat pixel average. This matters for photos
    with two very different color regions (e.g. half deep red, half
    white) — a flat average blurs those into a muddy color that
    represents neither; the dominant cluster picks an actual color that
    genuinely appears a lot in the image.
    """
    photo_kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
    labels = photo_kmeans.fit_predict(pixels_flat)
    counts = np.bincount(labels, minlength=k)
    dominant_idx = np.argmax(counts)
    return photo_kmeans.cluster_centers_[dominant_idx]  # [r, g, b], 0-1 range


def analyze_photo(photo_path: Path) -> np.ndarray:
    with Image.open(photo_path) as img:
        img = img.convert("RGB")
        img = img.resize(ANALYSIS_THUMBNAIL_SIZE)
        pixels = np.asarray(img, dtype=np.float32) / 255.0

    # Brightness and saturation stay as full-image averages — those are
    # legitimately meaningful as averages (overall exposure / vividness).
    brightness = (0.299 * pixels[:, :, 0] + 0.587 * pixels[:, :, 1] + 0.114 * pixels[:, :, 2]).mean()
    max_c = pixels.max(axis=2)
    min_c = pixels.min(axis=2)
    saturation = (max_c - min_c).mean()

    # Color and warmth come from the dominant color instead of a flat
    # average, so multicolor photos are represented by a color that
    # actually appears in them.
    avg_r, avg_g, avg_b = get_dominant_color(pixels.reshape(-1, 3))
    warmth = avg_r - avg_b

    return np.array([avg_r, avg_g, avg_b, brightness, saturation, warmth])


def find_best_cluster_count(scaled_matrix: np.ndarray, k_min: int = 2, k_max: int = 8) -> int:
    """
    Tries a range of cluster counts and picks the one with the best
    silhouette score (a measure of how well-separated and internally
    tight the clusters are — higher is better, ranges roughly -1 to 1).
    """
    n_samples = scaled_matrix.shape[0]
    k_max = min(k_max, n_samples - 1)  # silhouette needs at least 2 samples per cluster

    if k_max < k_min:
        fallback = max(2, min(k_min, n_samples))
        print(f"  ⚠ Not enough photos to try multiple cluster counts, using {fallback}.")
        return fallback

    best_k = k_min
    best_score = -1.0
    print("  Testing possible group counts:")
    for k in range(k_min, k_max + 1):
        trial_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        trial_labels = trial_kmeans.fit_predict(scaled_matrix)
        score = silhouette_score(scaled_matrix, trial_labels)
        print(f"    k={k}: silhouette score {score:.3f}")
        if score > best_score:
            best_score = score
            best_k = k

    print(f"  → Best number of aesthetic groups: {best_k} (silhouette score {best_score:.3f})")
    return best_k


def load_cache() -> dict:
    if not ENABLE_CACHE:
        return {}
    cache_path = Path(CACHE_FILE)
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}  # corrupted or unreadable cache — just start fresh


def save_cache(cache: dict) -> None:
    if not ENABLE_CACHE:
        return
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except OSError as e:
        print(f"  ⚠ Could not save analysis cache: {e}")


def analyze_photos_batch(photos: list[Path]) -> tuple[list[np.ndarray], list[Path]]:
    """
    Analyzes every photo's aesthetic features, skipping any that are
    already in the cache and unchanged since (same file size and modified
    time), and analyzing whatever's left in parallel across
    ANALYSIS_WORKERS threads.
    """
    cache = load_cache()
    features: list[np.ndarray | None] = [None] * len(photos)
    is_valid = [True] * len(photos)
    to_analyze: list[tuple[int, Path]] = []

    for i, photo in enumerate(photos):
        try:
            stat = photo.stat()
        except OSError:
            is_valid[i] = False
            continue

        key = str(photo.resolve())
        entry = cache.get(key)
        if entry and entry.get("mtime") == stat.st_mtime and entry.get("size") == stat.st_size:
            features[i] = np.array(entry["features"])
        else:
            to_analyze.append((i, photo))

    cache_hits = len(photos) - len(to_analyze)
    if cache_hits:
        print(f"  {cache_hits} photo(s) found in cache, skipping re-analysis.")

    if to_analyze:
        print(f"  Analyzing {len(to_analyze)} new/changed photo(s) using {ANALYSIS_WORKERS} workers...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=ANALYSIS_WORKERS) as executor:
            future_to_item = {executor.submit(analyze_photo, photo): (i, photo) for i, photo in to_analyze}
            for future in concurrent.futures.as_completed(future_to_item):
                i, photo = future_to_item[future]
                try:
                    feat = future.result()
                    features[i] = feat
                    stat = photo.stat()
                    cache[str(photo.resolve())] = {
                        "mtime": stat.st_mtime,
                        "size": stat.st_size,
                        "features": feat.tolist(),
                    }
                except Exception as e:
                    print(f"  ⚠ Skipping '{photo.name}': could not analyze it ({e})")
                    is_valid[i] = False

        save_cache(cache)

    valid_features = []
    valid_photos = []
    for i, photo in enumerate(photos):
        if is_valid[i] and features[i] is not None:
            valid_features.append(features[i])
            valid_photos.append(photo)

    return valid_features, valid_photos


def cluster_by_aesthetic(photos: list[Path], num_clusters: int | str):
    print(f"\nAnalyzing the visual aesthetic of {len(photos)} photos...")

    features, valid_photos = analyze_photos_batch(photos)

    if num_clusters != "auto" and len(valid_photos) < num_clusters:
        raise ValueError("Not enough photos to analyze. Reduce NUM_AESTHETIC_CLUSTERS.")

    feature_matrix = np.array(features)
    
    # Scale features so no single metric dominates Euclidean distance
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(feature_matrix)
    feature_by_photo = {photo: feat for photo, feat in zip(valid_photos, scaled_matrix)}

    if num_clusters == "auto":
        num_clusters = find_best_cluster_count(scaled_matrix)

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


_vividness_cache: dict[Path, float] = {}


def get_vividness_score(photo_path: Path) -> float:
    """
    A quick, standalone vividness (saturation) score used only for
    picking a cover photo — independent of the aesthetic-clustering
    pipeline so it works in both random and aesthetic mode. Cached since
    the same photo can show up across multiple patterns.
    """
    if photo_path in _vividness_cache:
        return _vividness_cache[photo_path]

    try:
        with Image.open(photo_path) as img:
            img = img.convert("RGB")
            img = img.resize(ANALYSIS_THUMBNAIL_SIZE)
            pixels = np.asarray(img, dtype=np.float32) / 255.0
        max_c = pixels.max(axis=2)
        min_c = pixels.min(axis=2)
        score = float((max_c - min_c).mean())
    except Exception:
        score = 0.0  # unreadable photo — don't crash, just never pick it as cover

    _vividness_cache[photo_path] = score
    return score


def apply_cover_photo(patterns: list[tuple[Path, ...]]) -> list[tuple[Path, ...]]:
    """
    Moves each pattern's most vivid (highest-saturation) photo to
    position 1 — since that's the first frame a TikTok viewer sees
    before swiping. The rest of the pattern keeps its existing order.
    """
    reordered = []
    for pattern in patterns:
        cover = max(pattern, key=get_vividness_score)
        rest = [p for p in pattern if p != cover]
        reordered.append((cover, *rest))
    return reordered


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


def sanitize_folder_name(name: str) -> str:
    """
    Makes a user-typed string safe to use as a single folder name.
    Replaces path separators (/ and \\) and other characters that are
    invalid or special on Windows/macOS/Linux filesystems, so something
    like "2026/2/15" becomes one folder called "2026-2-15" instead of
    three nested folders (2026 -> 2 -> 15).
    """
    invalid_chars = r'[\\/:*?"<>|]'
    sanitized = re.sub(invalid_chars, "-", name)
    sanitized = sanitized.strip().strip(".")  # trailing dots/spaces break on Windows
    return sanitized if sanitized else "batch"


def make_run_folder(output_folder: str) -> Path:
    """
    Asks the person what to call this batch, then creates exactly ONE
    folder named "<name>_1" directly inside OUTPUT_FOLDER, incrementing
    the number each time that name is reused (e.g. "cars_1", "cars_2",
    ...), so runs never collide or overwrite each other regardless of
    what name is picked.
    """
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    raw_name = input("What would you like to name this batch of patterns? ").strip()
    base_name = sanitize_folder_name(raw_name)

    n = 1
    while (output_path / f"{base_name}_{n}").exists():
        n += 1
    run_path = output_path / f"{base_name}_{n}"

    run_path.mkdir(parents=False, exist_ok=False)  # single folder only, never nested
    print(f"\nSaving this run's patterns to: {run_path}")
    return run_path


def run_random_mode():
    pool, folder_of = extract_pool(SOURCE_FOLDERS, min_total=PHOTOS_PER_PATTERN)
    patterns = generate_patterns(
        pool, NUM_PATTERNS, PHOTOS_PER_PATTERN,
        folder_of=folder_of, max_fraction_per_folder=MAX_FRACTION_PER_FOLDER,
    )
    if COVER_PHOTO_MODE:
        patterns = apply_cover_photo(patterns)
    run_path = make_run_folder(OUTPUT_FOLDER)
    save_patterns(patterns, str(run_path))


def run_aesthetic_mode():
    pool, folder_of = extract_pool(SOURCE_FOLDERS, min_total=PHOTOS_PER_PATTERN)
    clusters, kmeans, feature_by_photo, valid_photos = cluster_by_aesthetic(pool, NUM_AESTHETIC_CLUSTERS)

    run_path = make_run_folder(OUTPUT_FOLDER)

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
        if COVER_PHOTO_MODE:
            patterns = apply_cover_photo(patterns)
        save_patterns(patterns, str(run_path / cluster_name))


def print_config_summary():
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"  Mode: {MODE}")
    print(f"  Source folders ({len(SOURCE_FOLDERS)}):")
    for folder in SOURCE_FOLDERS:
        print(f"    - {folder}")
    print(f"  Output folder: {OUTPUT_FOLDER}")

    if MODE == "aesthetic":
        print(f"  Aesthetic clusters: {NUM_AESTHETIC_CLUSTERS}")
        print(f"  Patterns per cluster: {NUM_PATTERNS}")
    else:
        print(f"  Patterns to generate: {NUM_PATTERNS}")

    print(f"  Photos per pattern: {PHOTOS_PER_PATTERN}")
    print(f"  Max photos taken per folder: {MAX_FRACTION_PER_FOLDER:.0%}")

    if ORIENTATION_FILTER is not None or ASPECT_RATIO_FILTER is not None:
        parts = []
        if ORIENTATION_FILTER is not None:
            parts.append(f"orientation={ORIENTATION_FILTER}")
        if ASPECT_RATIO_FILTER is not None:
            parts.append(f"aspect ratio≈{ASPECT_RATIO_FILTER} (±{ASPECT_RATIO_TOLERANCE:.0%})")
        print(f"  Format filter: {', '.join(parts)}")
    else:
        print("  Format filter: none")

    print(f"  Cover-photo mode: {'ON (most vivid photo goes first)' if COVER_PHOTO_MODE else 'off'}")
    print("=" * 60)
    print()


def main():
    if ASPECT_RATIO_FILTER is not None:
        matches_aspect_ratio(16, 9, ASPECT_RATIO_FILTER, ASPECT_RATIO_TOLERANCE)  # validates the format early

    print_config_summary()

    if MODE == "random":
        run_random_mode()
    elif MODE == "aesthetic":
        run_aesthetic_mode()
    else:
        raise ValueError(f"Invalid MODE: '{MODE}'. Use 'random' or 'aesthetic'.")

if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as e:
        print(f"\n❌ {e}")