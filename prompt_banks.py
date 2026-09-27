"""Prompt-bank utilities with reproducible source-aware strategies.

This module separates three prompt-bank modes:
- ``legacy``: the manually curated banks we previously used in CTA/JPTA.
- ``published``: prompts traced to prompt-learning papers/repos such as
  CoOp, PromptSRC, ProDA/ProGrad, and APE/CuPL supplements.
- ``hybrid``: a union of the two, which is the strongest candidate for
  accuracy while still preserving the legacy bank for ablations.
"""

from __future__ import annotations

import os


LEGACY_TEMPLATE_BANKS = {
    "eurosat": [
        "a centered satellite photo of {}.",
        "a satellite photo of {}.",
        "an aerial photo of {}.",
        "satellite imagery of {}.",
        "an overhead view of {}.",
        "a remote sensing photo of {}.",
    ],
    "dtd": [
        "{} texture",
        "a {} texture",
        "a close-up of {} texture",
        "a detailed {} texture",
        "a photo of {} texture",
    ],
    "fgvc": [
        "a photo of a {}, a type of aircraft.",
        "a clear photo of a {} aircraft.",
        "a side view photo of a {} aircraft.",
        "a detailed photo of the aircraft {}.",
    ],
    "sun397": [
        "a photo of a {}.",
        "a photo of the {}.",
        "a photo of a {} scene.",
        "a photo of the {} scene.",
        "a scene of {}.",
        "a view of {}.",
    ],
    "stanford_cars": [
        "a photo of a {}.",
        "a photo of the {}.",
        "a photo of a {} car.",
        "a photo of the {} car.",
        "a close-up photo of a {} car.",
        "a detailed photo of a {} car.",
    ],
    "food101": [
        "a photo of {}, a type of food.",
        "a close-up photo of {}.",
        "a plated serving of {}.",
        "a photo of a dish of {}.",
        "a delicious serving of {}.",
    ],
    "oxford_flowers": [
        "a photo of a {}, a type of flower.",
        "a close-up photo of a {} flower.",
        "a photo of the {} flower.",
        "a blooming {} flower.",
        "a macro photo of a {} flower.",
    ],
    "caltech101": [
        "a photo of a {}.",
        "a photo of the {}.",
        "a centered photo of a {}.",
        "a close-up photo of a {}.",
        "an image of a {}.",
    ],
    "ucf101": [
        "a photo of a person doing {}.",
        "a photo of a person performing {}.",
        "a video frame of a person doing {}.",
        "a still image of a person performing {}.",
        "a photo of the action {}.",
    ],
}


# Published CLIP-style diversity prompts reused across CoOp / PromptSRC.
PUBLISHED_PHOTO_DIVERSITY_TEMPLATES = [
    "a photo of a {}.",
    "a photo of the {}.",
    "a close-up photo of a {}.",
    "a good photo of a {}.",
    "a bad photo of a {}.",
    "a blurry photo of a {}.",
    "a bright photo of a {}.",
    "a black and white photo of the {}.",
]


PUBLISHED_SCENE_DIVERSITY_TEMPLATES = [
    "a photo of a {}.",
    "a photo of the {}.",
    "a photo of a {} scene.",
    "a photo of the {} scene.",
    "a scene of {}.",
]


PUBLISHED_ACTION_DIVERSITY_TEMPLATES = [
    "a photo of a person doing {}.",
    "a photo of a person performing {}.",
    "a video frame of a person doing {}.",
    "a still image of a person doing {}.",
]


PUBLISHED_TEMPLATE_BANKS = {
    # CoOp / PromptSRC / APE / ProDA-style canonical templates.
    "eurosat": [
        "a centered satellite photo of {}.",
    ],
    "dtd": [
        "{} texture.",
        "a texture of {}.",
    ],
    "fgvc": [
        "a photo of a {}, a type of aircraft.",
        "a type of aircraft, a photo of a {}.",
    ],
    "sun397": [
        "a photo of a {}.",
        "Describe what a {} looks like",
    ],
    "stanford_cars": [
        "a photo of a {}.",
    ],
    "food101": [
        "a photo of {}, a type of food.",
        "a type of food, a photo of {}.",
    ],
    "oxford_flowers": [
        "a photo of a {}, a type of flower.",
        "a type of flower, a photo of a {}.",
    ],
    "caltech101": [
        "a photo of a {}.",
    ],
    "ucf101": [
        "a photo of a person doing {}.",
    ],
}


PUBLISHED_DIVERSITY_BY_DATASET = {
    "fgvc": PUBLISHED_PHOTO_DIVERSITY_TEMPLATES,
    "stanford_cars": PUBLISHED_PHOTO_DIVERSITY_TEMPLATES,
    "food101": PUBLISHED_PHOTO_DIVERSITY_TEMPLATES,
    "oxford_flowers": PUBLISHED_PHOTO_DIVERSITY_TEMPLATES,
    "caltech101": PUBLISHED_PHOTO_DIVERSITY_TEMPLATES,
    "sun397": PUBLISHED_SCENE_DIVERSITY_TEMPLATES,
    "ucf101": PUBLISHED_ACTION_DIVERSITY_TEMPLATES,
}


def _merge_unique_templates(*template_groups):
    merged = []
    seen = set()
    for group in template_groups:
        if not group:
            continue
        for template in group:
            if template not in seen:
                merged.append(template)
                seen.add(template)
    return merged


def _read_oxford_pets_species(dataset):
    species_map = {}
    metadata_path = os.path.join(dataset.anno_dir, "list.txt")
    with open(metadata_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            image_name, _, species_id, _ = line.split(" ")
            breed = "_".join(image_name.split("_")[:-1]).lower()
            species_map.setdefault(breed, "cat" if species_id == "1" else "dog")
    return species_map


def build_oxford_pets_template_bank(dataset, strategy="hybrid"):
    species_map = _read_oxford_pets_species(dataset)
    published_shared = [
        "a photo of a {}, a type of pet.",
        "a type of pet, a photo of a {}.",
    ]
    generic_shared = PUBLISHED_PHOTO_DIVERSITY_TEMPLATES

    template_bank = {}
    for classname in dataset.classnames:
        species = species_map.get(classname, "pet")
        legacy_templates = []
        if species == "cat":
            legacy_templates = [
                "a photo of a {}, a breed of cat.",
                "a clear photo of a {} cat.",
                "a close-up photo of a {} cat.",
                "a portrait of a {} cat.",
                "a household cat of breed {}.",
            ]
        elif species == "dog":
            legacy_templates = [
                "a photo of a {}, a breed of dog.",
                "a clear photo of a {} dog.",
                "a close-up photo of a {} dog.",
                "a portrait of a {} dog.",
                "a household dog of breed {}.",
            ]
        else:
            legacy_templates = ["a photo of a {}, a type of pet."]

        published_templates = _merge_unique_templates(published_shared, generic_shared)
        if strategy == "legacy":
            template_bank[classname] = legacy_templates
        elif strategy == "published":
            template_bank[classname] = published_templates
        else:
            template_bank[classname] = _merge_unique_templates(legacy_templates, published_templates)
    return template_bank


def get_template_bank(dataset_tag, dataset=None, strategy="hybrid"):
    if dataset_tag == "oxford_pets":
        if dataset is None:
            raise ValueError("Oxford Pets prompt banks require the dataset object.")
        return build_oxford_pets_template_bank(dataset, strategy=strategy)

    legacy_templates = LEGACY_TEMPLATE_BANKS.get(dataset_tag)
    published_templates = _merge_unique_templates(
        PUBLISHED_TEMPLATE_BANKS.get(dataset_tag),
        PUBLISHED_DIVERSITY_BY_DATASET.get(dataset_tag),
    )

    if strategy == "legacy":
        return legacy_templates
    if strategy == "published":
        return published_templates
    return _merge_unique_templates(legacy_templates, published_templates)
