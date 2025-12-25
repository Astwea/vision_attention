"""Dataset factory for unified dataset loading interface."""

from typing import Tuple, Optional, Dict
from torch.utils.data import DataLoader

from .cifar import get_cifar_dataloaders
from .imagenet import get_imagenet_dataloaders
from .coco import get_coco_dataloaders
from .voc import get_voc_dataloaders


def get_dataloaders(
    dataset_name: str,
    task_type: str,
    root: str,
    input_size: int,
    patch_grid_size: int,
    batch_size: int,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Unified interface for getting dataloaders for different datasets and tasks.
    
    Args:
        dataset_name: Name of dataset ("cifar10", "cifar100", "imagenet", "coco", "voc")
        task_type: Type of task ("classification", "detection", "segmentation", "instance_segmentation")
        root: Root directory for dataset
        input_size: Input image size
        patch_grid_size: Patch grid size
        batch_size: Batch size
        num_workers: Number of data loading workers
        **kwargs: Additional dataset-specific arguments
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        test_loader may be None if dataset doesn't have separate test set
    """
    dataset_name = dataset_name.lower()
    task_type = task_type.lower()
    
    # Map task types
    if task_type == "classification":
        coco_task_type = "detection"  # Not used for classification
        voc_task_type = "detection"  # Not used for classification
    elif task_type == "detection":
        coco_task_type = "detection"
        voc_task_type = "detection"
    elif task_type == "segmentation":
        coco_task_type = "instance_segmentation"
        voc_task_type = "segmentation"
    elif task_type == "instance_segmentation":
        coco_task_type = "instance_segmentation"
        voc_task_type = "segmentation"
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    
    if dataset_name in ["cifar10", "cifar100"]:
        if task_type != "classification":
            raise ValueError(f"Dataset {dataset_name} only supports classification task")
        
        train_loader, val_loader, test_loader = get_cifar_dataloaders(
            root=kwargs.get('dataset_root', root),
            dataset_name=dataset_name,
            input_size=input_size,
            patch_grid_size=patch_grid_size,
            batch_size=batch_size,
            num_workers=num_workers,
            download=kwargs.get('download', True),
        )
        return train_loader, val_loader, test_loader
    
    elif dataset_name == "imagenet":
        if task_type != "classification":
            raise ValueError(f"Dataset {dataset_name} only supports classification task")
        
        train_loader, val_loader = get_imagenet_dataloaders(
            root=kwargs.get('dataset_root', root),
            input_size=input_size,
            patch_grid_size=patch_grid_size,
            batch_size=batch_size,
            num_workers=num_workers,
            download=kwargs.get('download', False),
        )
        return train_loader, val_loader, None
    
    elif dataset_name == "coco":
        train_loader, val_loader = get_coco_dataloaders(
            root=kwargs.get('dataset_root', root),
            ann_file_train=kwargs.get('ann_file_train', "annotations/instances_train2017.json"),
            ann_file_val=kwargs.get('ann_file_val', "annotations/instances_val2017.json"),
            task_type=coco_task_type,
            input_size=input_size,
            patch_grid_size=patch_grid_size,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        return train_loader, val_loader, None
    
    elif dataset_name == "voc" or dataset_name == "pascal_voc":
        train_loader, val_loader = get_voc_dataloaders(
            root=kwargs.get('dataset_root', root),
            year=kwargs.get('year', "2012"),
            task_type=voc_task_type,
            input_size=input_size,
            patch_grid_size=patch_grid_size,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        return train_loader, val_loader, None
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_dataset_info(dataset_name: str, task_type: str) -> Dict:
    """
    Get dataset information (number of classes, etc.).
    
    Args:
        dataset_name: Name of dataset
        task_type: Type of task
        
    Returns:
        Dictionary with dataset information
    """
    dataset_name = dataset_name.lower()
    task_type = task_type.lower()
    
    info = {}
    
    if dataset_name in ["cifar10", "cifar100"]:
        info['num_classes'] = 10 if dataset_name == "cifar10" else 100
        info['task_type'] = "classification"
    
    elif dataset_name == "imagenet":
        info['num_classes'] = 1000
        info['task_type'] = "classification"
    
    elif dataset_name == "coco":
        if task_type in ["detection", "classification"]:
            info['num_classes'] = 80
        elif task_type in ["segmentation", "instance_segmentation"]:
            info['num_classes'] = 80  # Same 80 classes for instance segmentation
        info['task_type'] = task_type
    
    elif dataset_name in ["voc", "pascal_voc"]:
        if task_type == "detection":
            info['num_classes'] = 20  # 20 object classes (excluding background)
        elif task_type == "segmentation":
            info['num_classes'] = 21  # 20 object classes + background
        info['task_type'] = task_type
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return info

