# mmdet/datasets/dior_increment.py
from mmdet.registry import DATASETS
from .coco_increment import CocoIncDataset


@DATASETS.register_module()
class DotaIncDataset(CocoIncDataset):
    METAINFO = {
        'classes':
            ("small-vehicle","large-vehicle","airplane","baseball-diamond","ground-track-field","helicopter","ship","bridge","soccer-ball-field","tennis-court",
            "storage-tank","harbor","roundabout","basketball-court","swimming-pool"
    ),
        'id': (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14)
    }

    def __init__(self, start: int = 0, end: int = 15, **kwargs) -> None:
        super().__init__(start=start, end=end, **kwargs)