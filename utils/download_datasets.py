"""Utility script to download datasets automatically."""

import os
import sys
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from typing import Optional


def download_file(url: str, destination: str, description: str = ""):
    """Download a file with progress bar."""
    print(f"下载 {description}...")
    print(f"URL: {url}")
    print(f"目标: {destination}")
    
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r进度: {percent}%")
        sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, destination, reporthook=progress_hook)
        print("\n下载完成!")
        return True
    except Exception as e:
        print(f"\n下载失败: {e}")
        return False


def extract_zip(zip_path: str, extract_to: str):
    """Extract zip file."""
    print(f"解压 {zip_path} 到 {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("解压完成!")


def extract_tar(tar_path: str, extract_to: str):
    """Extract tar file."""
    print(f"解压 {tar_path} 到 {extract_to}...")
    with tarfile.open(tar_path, 'r:*') as tar_ref:
        tar_ref.extractall(extract_to)
    print("解压完成!")


def download_coco(root: str = "./data/coco"):
    """Download COCO dataset."""
    print("=" * 60)
    print("下载 COCO 数据集")
    print("=" * 60)
    
    os.makedirs(root, exist_ok=True)
    
    base_url = "http://images.cocodataset.org"
    
    # Files to download
    files = [
        ("zips/train2017.zip", "train2017.zip"),
        ("zips/val2017.zip", "val2017.zip"),
        ("annotations/annotations_trainval2017.zip", "annotations_trainval2017.zip"),
    ]
    
    for url_path, filename in files:
        file_path = os.path.join(root, filename)
        url = f"{base_url}/{url_path}"
        
        if os.path.exists(file_path):
            print(f"{filename} 已存在，跳过下载")
            continue
        
        if download_file(url, file_path, filename):
            # Extract if it's a zip file
            if filename.endswith('.zip'):
                if 'annotations' in filename:
                    extract_to = root
                elif 'train' in filename or 'val' in filename:
                    extract_to = root
                else:
                    extract_to = root
                extract_zip(file_path, extract_to)
                # Remove zip file after extraction
                os.remove(file_path)
                print(f"已删除 {filename}")
    
    print("\nCOCO 数据集下载完成!")
    print(f"数据集路径: {root}")


def download_voc(root: str = "./data/voc", year: str = "2012"):
    """Download Pascal VOC dataset."""
    print("=" * 60)
    print(f"下载 Pascal VOC {year} 数据集")
    print("=" * 60)
    
    os.makedirs(root, exist_ok=True)
    
    # Pascal VOC download URLs
    base_url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012"
    
    if year == "2012":
        url = f"{base_url}/VOCtrainval_11-May-2012.tar"
    elif year == "2007":
        url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"
    else:
        print(f"不支持的年份: {year}")
        return
    
    filename = url.split('/')[-1]
    file_path = os.path.join(root, filename)
    
    if os.path.exists(file_path):
        print(f"{filename} 已存在，跳过下载")
    else:
        if download_file(url, file_path, filename):
            extract_tar(file_path, root)
            # Remove tar file after extraction
            os.remove(file_path)
            print(f"已删除 {filename}")
    
    print(f"\nPascal VOC {year} 数据集下载完成!")
    print(f"数据集路径: {root}")


def download_imagenet(root: str = "./data/imagenet"):
    """Download ImageNet dataset - Note: requires manual registration."""
    print("=" * 60)
    print("ImageNet 数据集下载")
    print("=" * 60)
    print("\n注意: ImageNet 数据集需要手动下载!")
    print("原因: ImageNet 数据集需要注册并接受使用条款")
    print("\n请按照以下步骤操作:")
    print("1. 访问 https://www.image-net.org/download.php")
    print("2. 注册账号并登录")
    print("3. 下载 ILSVRC2012_img_train.tar 和 ILSVRC2012_img_val.tar")
    print("4. 将文件解压到以下目录:")
    print(f"   {root}/train/")
    print(f"   {root}/val/")
    print("\n或者使用以下命令 (如果有镜像链接):")
    print("# wget [train.tar链接] -O train.tar")
    print("# wget [val.tar链接] -O val.tar")
    print("# tar -xf train.tar -C", root)
    print("# tar -xf val.tar -C", root)
    print("\n下载完成后，请运行以下命令解压:")
    print(f"cd {root}/train")
    print("find . -name \"*.tar\" | while read NAME ; do mkdir -p \"${NAME%.tar}\"; tar -xf \"${NAME}\" -C \"${NAME%.tar}\"; rm -f \"${NAME}\"; done")


def main():
    """Main function to download all datasets."""
    import argparse
    
    parser = argparse.ArgumentParser(description='下载数据集')
    parser.add_argument('--dataset', type=str, choices=['coco', 'voc', 'imagenet', 'all'],
                       default='all', help='要下载的数据集')
    parser.add_argument('--root', type=str, default='./data', help='数据集根目录')
    parser.add_argument('--voc-year', type=str, default='2012', choices=['2007', '2012'],
                       help='Pascal VOC 年份')
    
    args = parser.parse_args()
    
    if args.dataset == 'coco' or args.dataset == 'all':
        coco_root = os.path.join(args.root, 'coco')
        download_coco(coco_root)
    
    if args.dataset == 'voc' or args.dataset == 'all':
        voc_root = os.path.join(args.root, 'voc')
        download_voc(voc_root, args.voc_year)
    
    if args.dataset == 'imagenet' or args.dataset == 'all':
        imagenet_root = os.path.join(args.root, 'imagenet')
        download_imagenet(imagenet_root)
    
    print("\n" + "=" * 60)
    print("所有数据集下载完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()

