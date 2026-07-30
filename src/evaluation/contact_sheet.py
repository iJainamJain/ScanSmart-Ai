"""Build a labelled montage of pipeline results.

Reviewing detection quality one image at a time is slow enough that it
invites shortcuts - and spot-checking a handful is exactly how a change was
once accepted that turned out to be a net regression across the full set.
A contact sheet makes reviewing every image a single glance, so full-batch
visual verification stays cheap.
"""

from pathlib import Path

import cv2
import numpy as np


def build_contact_sheet(
    images: list[tuple[str, np.ndarray]],
    output_path: Path,
    columns: int = 8,
    cell: int = 300,
) -> Path:
    """Tile labelled thumbnails into one image and write it to output_path."""
    cells = []
    for label, image in images:
        if image is None:
            continue
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        height, width = image.shape[:2]
        scale = min(cell / width, (cell - 26) / height)
        thumb = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))))

        tile = np.full((cell, cell, 3), 40, np.uint8)
        y = 26 + ((cell - 26) - thumb.shape[0]) // 2
        x = (cell - thumb.shape[1]) // 2
        tile[y:y + thumb.shape[0], x:x + thumb.shape[1]] = thumb
        cv2.putText(tile, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cells.append(tile)

    if not cells:
        raise ValueError("no images to tile")

    blank = np.full((cell, cell, 3), 40, np.uint8)
    rows = []
    for start in range(0, len(cells), columns):
        row = cells[start:start + columns]
        row += [blank] * (columns - len(row))
        rows.append(np.hstack(row))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.vstack(rows))
    return output_path
