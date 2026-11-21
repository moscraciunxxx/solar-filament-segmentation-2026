"""Annotator sampling (LS): on every augmented load of a multi-annotator train image, use one annotator's eroded
labels chosen uniformly at random (python `random`, seeded per dataloader worker by ultralytics), instead of the fixed
eroded medoid. Single-annotator images and non-augmented (val) loads are unchanged. Parsing mirrors ultralytics
verify_image_label for segment labels (class + polygon -> segments2boxes, duplicate-row removal)."""
import json, random
from pathlib import Path
import numpy as np
from ultralytics.data import base as _base
from ultralytics.utils.ops import segments2boxes
STATS = {"calls": 0, "swapped": 0}
def parse(text):
    lb = [x.split() for x in text.strip().splitlines() if x.strip()]
    if not lb:
        return {"cls": np.zeros((0, 1), np.float32), "bboxes": np.zeros((0, 4), np.float32), "segments": []}
    classes = np.array([x[0] for x in lb], dtype=np.float32)
    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]
    arr = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1).astype(np.float32)
    _, i = np.unique(arr, axis=0, return_index=True)
    if len(i) < len(arr):
        arr = arr[i]; segments = [segments[x] for x in i]
    return {"cls": arr[:, 0:1], "bboxes": arr[:, 1:], "segments": segments}
def install(json_path):
    data = json.load(open(json_path))
    alt = {s: [parse(t) for t in d["texts"]] for s, d in data.items() if len(d["texts"]) > 1}
    orig = _base.BaseDataset.get_image_and_label
    def get_image_and_label(self, index):
        if getattr(self, "augment", False):
            STATS["calls"] += 1
            alts = alt.get(Path(self.labels[index]["im_file"]).stem)
            if alts:
                a = alts[random.randrange(len(alts))]
                saved = self.labels[index]
                tmp = dict(saved)
                tmp.update(cls=a["cls"].copy(), bboxes=a["bboxes"].copy(), segments=[s.copy() for s in a["segments"]])
                self.labels[index] = tmp; STATS["swapped"] += 1
                try:
                    return orig(self, index)
                finally:
                    self.labels[index] = saved
        return orig(self, index)
    _base.BaseDataset.get_image_and_label = get_image_and_label
    return len(alt)
