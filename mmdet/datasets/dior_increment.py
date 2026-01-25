# mmdet/datasets/dior_increment.py
from mmdet.registry import DATASETS
from .coco_increment import CocoIncDataset


@DATASETS.register_module()
class DiorIncDataset(CocoIncDataset):
    METAINFO = {
        'classes':
            ("airplane",
    "airport",
    "bridge",
    "Expressway-Service-area",
    "Expressway-toll-station",
    "harbor",
    "overpass",
    "ship",
    "trainstation",
    "vehicle",
    "baseballfield",
    "basketballcourt",
    "chimney",
    "dam",
    "golffield",
    "groundtrackfield",
    "stadium",
    "storagetank",
    "tenniscourt",
    "windmill"),
        'id': tuple(range(0, 20))
    }

    def __init__(self, start: int = 0, end: int = 20, **kwargs) -> None:
        super().__init__(start=start, end=end, **kwargs)