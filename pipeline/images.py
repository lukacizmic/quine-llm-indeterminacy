"""Folder-of-images -> base64-encoded image list.

Each concept lives in its own folder under ``50_things_images/<concept>/``.
Every trial should use the same, fixed number of images per concept so that
prompt length and evidential context are comparable across concepts.
"""
from __future__ import annotations

import base64
import os
import random
from dataclasses import dataclass

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


@dataclass(frozen=True)
class LoadedImages:
    concept: str
    folder_path: str
    filenames: list[str]  # the filenames actually used, in the order used
    encoded: list[str]  # base64-encoded bytes, same order as filenames


def list_image_files(folder_path: str) -> list[str]:
    """Return sorted image filenames in a concept folder (stable, reproducible order)."""
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Concept image folder not found: {folder_path}")
    files = [
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]
    return sorted(files)


def encode_image(filepath: str) -> str:
    with open(filepath, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def load_concept_images(
    images_root: str,
    concept: str,
    max_images: int = 10,
    shuffle_seed: int | None = None,
) -> LoadedImages:
    """Load and base64-encode up to ``max_images`` images for a concept.

    Parameters
    ----------
    images_root: root directory containing one subfolder per concept
        (e.g. ``50_things_images``).
    concept: the concept/folder name (must match the folder exactly).
    max_images: fixed cap on images per trial so every concept contributes
        the same amount of evidence. If a folder has fewer images, all of
        them are used (this should be flagged/logged by the caller).
    shuffle_seed: if provided, images are shuffled deterministically with
        this seed before truncation instead of using plain sorted order.
        The resulting filename order is always returned so it can be logged
        per trial for reproducibility.
    """
    folder_path = os.path.join(images_root, concept)
    filenames = list_image_files(folder_path)

    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        filenames = filenames[:]
        rng.shuffle(filenames)

    selected = filenames[:max_images]
    encoded = [encode_image(os.path.join(folder_path, f)) for f in selected]

    return LoadedImages(
        concept=concept,
        folder_path=folder_path,
        filenames=selected,
        encoded=encoded,
    )


def list_available_concepts(images_root: str) -> list[str]:
    """Return sorted concept folder names available under images_root."""
    return sorted(
        name
        for name in os.listdir(images_root)
        if os.path.isdir(os.path.join(images_root, name))
    )
