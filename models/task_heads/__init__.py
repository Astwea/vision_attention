"""Task heads for different vision tasks."""

from .classification_head import ClassificationHead
from .detection_head import DetectionHead
from .segmentation_head import SegmentationHead, InstanceSegmentationHead

__all__ = [
    'ClassificationHead',
    'DetectionHead',
    'SegmentationHead',
    'InstanceSegmentationHead',
]

