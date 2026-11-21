"""Ground truth limited to the medoid annotator of one JPEG.

The official score uses every annotator. This view is the single outline
the medoid rule would draw. It does not read test labels.
"""

from __future__ import annotations

from .coco_io import AnnotatorView, medoid_view_id, rasterize_polygons
from .rle import encode_rle


def medoid_gt_records(
    views: dict[str, AnnotatorView],
    image_ids: list[str],
) -> list[dict]:
    """One annotator's filaments: the medoid view, never a stack of raters."""
    view = views[medoid_view_id(views, image_ids)]
    records = []
    for index, filament in enumerate(view.filaments, start=1):
        mask = rasterize_polygons([filament.polygon])
        records.append(
            {
                "annotator_image": view.image_id,
                "index": index,
                "rle": encode_rle(mask),
                "file_name": view.file_name,
            }
        )
    return records
