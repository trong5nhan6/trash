"""Text prompts for the vision-language zero-shot head (method M2).

'plain'      : class name in generic templates (CLIP-style baseline).
'definition' : what each recycling category *contains*, written from general municipal recycling
               guidelines (material stream definitions), NOT from inspecting the test images.
               Fixed before any evaluation. Each description -> one prompt; class weight = mean.
"""
from common import CORE

PLAIN_TEMPLATES = [
    "a photo of {}.",
    "a photo of {}, a type of waste.",
    "a photo of discarded {}.",
    "a photo of {} trash for recycling.",
]

DEFINITIONS = {
    "cardboard": [
        "a corrugated cardboard box",
        "flattened brown cardboard",
        "a paperboard box such as a cereal box or a shoe box",
        "printed paperboard food packaging",
        "a crumpled cardboard package",
        "a cardboard tube",
        "a pizza box",
        "an egg carton made of cardboard",
    ],
    "glass": [
        "a glass bottle",
        "a clear glass jar",
        "a green or brown glass beer bottle",
        "a wine bottle",
        "broken glass",
        "a drinking glass",
    ],
    "metal": [
        "an aluminum drink can",
        "a steel food can",
        "an aerosol spray can",
        "aluminum foil or an aluminum tray",
        "a metal lid or bottle cap",
        "a crushed metal can",
        "scrap metal",
    ],
    "paper": [
        "a sheet of office paper",
        "a newspaper",
        "a magazine",
        "crumpled paper",
        "a paper bag",
        "an envelope or junk mail",
        "shredded paper",
    ],
    "plastic": [
        "a plastic bottle",
        "a plastic food container",
        "a plastic bag",
        "plastic film wrap",
        "a plastic cup",
        "a crushed plastic bottle",
        "a plastic clamshell package",
    ],
}
DEF_TEMPLATES = ["a photo of {}.", "a photo of {}, discarded as waste."]


def prompt_sets():
    """-> {set_name: [[prompts for class 0], [prompts for class 1], ...]} in CORE order."""
    return {
        "plain": [[t.format(c) for t in PLAIN_TEMPLATES] for c in CORE],
        "definition": [[t.format(d) for d in DEFINITIONS[c] for t in DEF_TEMPLATES] for c in CORE],
    }
