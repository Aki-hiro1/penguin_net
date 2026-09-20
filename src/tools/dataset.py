from tools.calculate_tool import indices_generate
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from rasterio import features
import geopandas as gpd
import random
from scipy.ndimage import distance_transform_edt


class ImgIdxDataset(Dataset):
    """
    遥感影像 + 点标签 数据集
    从 load_scenes 返回的 patches 列表加载数据
    实时计算13个指数通道
    支持数据增强（手动实现）
    """

    def __init__(self, patches, patch_size=128, augment=True, subset_indices=None):
        """
        Args:
            patches: load_scenes 返回的图幅列表
            patch_size: 图幅尺寸
            augment: 是否启用数据增强
            subset_indices: 子集索引（用于划分训练/验证/测试）
        """
        self.patch_size = patch_size
        self.augment = augment

        if subset_indices is not None:
            self.patches = [patches[i] for i in subset_indices]
        else:
            self.patches = patches

        print(f"加载完成: {len(self.patches)} 个图幅")

    def _apply_augmentations(self, img, indices, label, mask):
        """
        手动实现数据增强（同时对 img、indices、label、mask 做相同的几何变换）
        仅保留几何变换（翻转、旋转），删除光度变换（亮度/对比度/噪声）
        """
        H, W = img.shape[1], img.shape[2]

        # 1. 水平翻转
        if random.random() < 0.5:
            img = np.flip(img, axis=2).copy()
            indices = np.flip(indices, axis=2).copy()
            label = np.flip(label, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()  # mask同步翻转

        # 2. 垂直翻转
        if random.random() < 0.3:
            img = np.flip(img, axis=1).copy()
            indices = np.flip(indices, axis=1).copy()
            label = np.flip(label, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()  # mask同步翻转

        # 3. 随机旋转 90/180/270 度
        if random.random() < 0.5:
            k = random.choice([1, 2, 3])
            img = np.rot90(img, k, axes=(1, 2)).copy()
            indices = np.rot90(indices, k, axes=(1, 2)).copy()
            label = np.rot90(label, k, axes=(0, 1)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()  # mask同步旋转

        return img, indices, label, mask  # 返回mask

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]

        # 1. 提取数据
        img = patch['image_data']  # [6, H, W]
        points = patch['points']
        H, W = img.shape[1], img.shape[2]

        # 2. 实时计算13个指数 [13, H, W]
        indices = indices_generate(img)

        # 3. 生成标签图
        label = np.zeros((H, W), dtype=np.int64)
        rows = points['rows']
        cols = points['cols']
        lbls = points['labels']
        for r, c, l in zip(rows, cols, lbls):
            if 0 <= r < H and 0 <= c < W:
                label[r, c] = int(l)

        # 4. 生成损失掩码
        mask = np.zeros((H, W), dtype=np.float32)
        mask[rows, cols] = 1.0

        # 5. 裁剪到 patch_size
        if H != self.patch_size or W != self.patch_size:
            top = np.random.randint(0, H - self.patch_size + 1)
            left = np.random.randint(0, W - self.patch_size + 1)
            img = img[:, top:top+self.patch_size, left:left+self.patch_size]
            indices = indices[:, top:top+self.patch_size,
                              left:left+self.patch_size]
            label = label[top:top+self.patch_size, left:left+self.patch_size]
            mask = mask[top:top+self.patch_size, left:left+self.patch_size]

        # 6. 数据增强（仅几何变换，mask同步变换）
        if self.augment:
            img, indices, label, mask = self._apply_augmentations(  # ⬅️ 传入并接收mask
                img, indices, label, mask
            )

        # 7. 转为 Tensor（关键：转换为 float32）
        img = torch.from_numpy(img.astype(np.float32))
        indices = torch.from_numpy(indices.astype(np.float32))
        label = torch.from_numpy(label).long()
        mask = torch.from_numpy(mask)

        return {
            'image': img,
            'indices': indices,
            'label': label,
            'mask': mask,
            'scene_id': patch.get('scene_id', 'unknown')
        }


# ========== 测试 ==========
if __name__ == "__main__":
    # 这里需要先有 patches 才能测试
    from tools.file_tool import load_scenes
    patches = load_scenes("./config/context_manifest.yaml", 128)

    dataset = ImgIdxDataset(
        patches=patches,
        patch_size=128,
        augment=True
    )
    sample = dataset[0]
    print(f"Image: {sample['image'].shape}")
    print(f"Indices: {sample['indices'].shape}")
    print(f"Label: {sample['label'].shape}")
    print(f"Mask: {sample['mask'].shape}")

    # 验证mask和label是否对齐
    label_np = sample['label'].numpy()
    mask_np = sample['mask'].numpy()
    print(f"Label中非零位置: {np.where(label_np != 0)}")
    print(f"Mask中为1的位置: {np.where(mask_np == 1)}")
