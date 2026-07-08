from __future__ import annotations

from pathlib import Path

from model.defect_info import DefectInfo


class BBoxDrawer:
    def draw(self, image: object, defects: list[DefectInfo], output_path: Path) -> Path:
        # TODO: Draw defect bounding boxes and write the result image.
        return output_path
