"""Datasets package for vision attention routing framework."""

# Optional imports - only import modules that exist
try:
    from .cifar import get_cifar_dataloaders, CIFARPatchDataset
except ImportError:
    get_cifar_dataloaders = None
    CIFARPatchDataset = None

try:
    from .imagenet import get_imagenet_dataloaders, ImageNetPatchDataset
except ImportError:
    get_imagenet_dataloaders = None
    ImageNetPatchDataset = None

try:
    from .coco import get_coco_dataloaders, COCODetectionDataset, COCOInstanceSegDataset
except ImportError:
    get_coco_dataloaders = None
    COCODetectionDataset = None
    COCOInstanceSegDataset = None

try:
    from .voc import get_voc_dataloaders, VOCDetectionDataset, VOCSegmentationDataset
except ImportError:
    get_voc_dataloaders = None
    VOCDetectionDataset = None
    VOCSegmentationDataset = None

from .cluttered_mnist import get_cluttered_mnist_dataloaders, ClutteredMNIST
from .factory import get_dataloaders, get_dataset_info

__all__ = [
    'get_cluttered_mnist_dataloaders',
    'ClutteredMNIST',
    'get_dataloaders',
    'get_dataset_info',
]

# Add optional exports if they exist
if get_cifar_dataloaders is not None:
    __all__.extend(['get_cifar_dataloaders', 'CIFARPatchDataset'])
if get_imagenet_dataloaders is not None:
    __all__.extend(['get_imagenet_dataloaders', 'ImageNetPatchDataset'])
if get_coco_dataloaders is not None:
    __all__.extend(['get_coco_dataloaders', 'COCODetectionDataset', 'COCOInstanceSegDataset'])
if get_voc_dataloaders is not None:
    __all__.extend(['get_voc_dataloaders', 'VOCDetectionDataset', 'VOCSegmentationDataset'])

