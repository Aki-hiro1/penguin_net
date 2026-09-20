import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import GroupKFold

# 路径设置
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from tools.file_tool import load_scenes
    from tools.dataset import ImgIdxDataset
    from models.unet_model import UNet
    from models.se_unet_model import SE_UNet
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)


# ========== 配置 ==========
class Config:
    # 数据路径
    MANIFEST_PATH = "./config/context_manifest.yaml"
    PATCH_SIZE = 128

    # 模型参数
    IN_CHANNELS_A = 6
    IN_CHANNELS_B = 13
    NUM_CLASSES = 3
    FEATURES = [64, 128, 256, 512]
    SE_REDUCTION = 16

    # 训练参数
    BATCH_SIZE = 8
    EPOCHS = 5
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 4

    # 数据划分
    N_FOLDS = 5          # GroupKFold 的 fold 数
    VAL_RATIO = 0.2      # 从训练集中再切出验证集的比例

    # 损失函数
    IGNORE_INDEX = 2

    # 保存路径
    SAVE_DIR = "./checkpoints"

    # 设备
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_loss_with_mask(logits, labels, mask, ignore_index=2):
    """带掩码的损失计算（点监督）"""
    mask = mask.clone()
    mask[labels == ignore_index] = 0

    B, C, H, W = logits.shape
    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
    labels_flat = labels.reshape(-1)
    mask_flat = mask.reshape(-1)

    valid_indices = mask_flat > 0.5
    if valid_indices.sum() == 0:
        return torch.tensor(0.0, device=logits.device)

    logits_valid = logits_flat[valid_indices]
    labels_valid = labels_flat[valid_indices]

    loss = nn.functional.cross_entropy(logits_valid, labels_valid)
    return loss


def validate(model, val_loader, config):
    """验证"""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="验证中"):
            img = batch['image'].to(config.DEVICE)
            indices = batch['indices'].to(config.DEVICE)
            label = batch['label'].to(config.DEVICE)
            mask = batch['mask'].to(config.DEVICE)

            logits = model(img, indices)
            loss = compute_loss_with_mask(
                logits, label, mask, config.IGNORE_INDEX)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0


def save_checkpoint(state, filename):
    """保存检查点"""
    torch.save(state, filename)
    print(f"检查点已保存: {filename}")


def create_data_loaders(patches, config):
    """
    按场景（scene_id）划分数据，保证同一 shp 来源的图幅
    不会同时出现在训练和验证/测试中。
    """
    groups = np.array([p['scene_id'] for p in patches])
    unique_scenes = np.unique(groups)

    print(f"总图幅数: {len(patches)}")
    print(f"不同场景数: {len(unique_scenes)}")
    print(f"场景列表: {unique_scenes.tolist()}")

    n_splits = min(config.N_FOLDS, len(unique_scenes))
    if n_splits < 2:
        raise ValueError(f"场景数不足（{len(unique_scenes)}），无法划分训练/验证/测试")

    gkf = GroupKFold(n_splits=n_splits)

    # 取第一个 fold 作为 训练+验证 / 测试 的划分
    fold_list = list(gkf.split(patches, groups=groups))
    train_val_idx, test_idx = fold_list[0]

    # 从 train_val 中再切出验证集
    train_val_groups = groups[train_val_idx]
    n_train_val_scenes = len(np.unique(train_val_groups))

    if n_train_val_scenes >= 2:
        n_inner = min(3, n_train_val_scenes)
        gkf_inner = GroupKFold(n_splits=n_inner)
        inner_list = list(gkf_inner.split(
            train_val_idx, groups=train_val_groups))
        train_inner, val_inner = inner_list[0]
        train_idx = train_val_idx[train_inner]
        val_idx = train_val_idx[val_inner]
    else:
        # 场景太少，无法再切验证集，用测试集当验证集
        train_idx = train_val_idx
        val_idx = test_idx

    train_scenes = set(patches[i]['scene_id'] for i in train_idx)
    val_scenes = set(patches[i]['scene_id'] for i in val_idx)
    test_scenes = set(patches[i]['scene_id'] for i in test_idx)

    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_idx)} 图幅, 场景: {sorted(train_scenes)}")
    print(f"  验证集: {len(val_idx)} 图幅, 场景: {sorted(val_scenes)}")
    print(f"  测试集: {len(test_idx)} 图幅, 场景: {sorted(test_scenes)}")

    # 检查场景无重叠
    assert len(train_scenes & val_scenes) == 0, "训练集和验证集有场景重叠"
    assert len(train_scenes & test_scenes) == 0, "训练集和测试集有场景重叠"
    assert len(val_scenes & test_scenes) == 0, "验证集和测试集有场景重叠"
    print("  场景无重叠检查通过")

    # 训练集开启增强，验证集和测试集不增强
    train_dataset = ImgIdxDataset(
        patches=patches,
        patch_size=config.PATCH_SIZE,
        augment=True,
        subset_indices=train_idx
    )

    val_dataset = ImgIdxDataset(
        patches=patches,
        patch_size=config.PATCH_SIZE,
        augment=False,
        subset_indices=val_idx
    )

    test_dataset = ImgIdxDataset(
        patches=patches,
        patch_size=config.PATCH_SIZE,
        augment=False,
        subset_indices=test_idx
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def train():
    """主训练函数"""
    config = Config()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("训练启动")
    print(f"设备: {config.DEVICE}")
    print(f"配置: {vars(config)}")

    # 创建保存目录
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    # 加载数据
    print("\n加载数据...")
    patches = load_scenes(config.MANIFEST_PATH, config.PATCH_SIZE, sample_num=[
                          300, 200, 200, 200, 100, 300, 60, 180])
    print(f"共加载 {len(patches)} 个图幅")

    train_loader, val_loader, test_loader = create_data_loaders(
        patches, config)

    # 创建模型
    print("\n创建模型...")
    model = SE_UNet(
        in_channels_a=config.IN_CHANNELS_A,
        in_channels_b=config.IN_CHANNELS_B,
        num_classes=config.NUM_CLASSES,
        features=config.FEATURES,
        se_reduction=config.SE_REDUCTION
    ).to(config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,} (可训练: {trainable_params:,})")

    # 优化器和调度器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.EPOCHS,
        eta_min=config.LEARNING_RATE * 0.01
    )

    # 训练
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(1, config.EPOCHS + 1):
        # 训练
        model.train()
        epoch_loss = 0
        num_batches = 0

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch}/{config.EPOCHS}")
        for batch in progress_bar:
            img = batch['image'].to(config.DEVICE)
            indices = batch['indices'].to(config.DEVICE)
            label = batch['label'].to(config.DEVICE)
            mask = batch['mask'].to(config.DEVICE)

            logits = model(img, indices)
            loss = compute_loss_with_mask(
                logits, label, mask, config.IGNORE_INDEX)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0

        # 验证
        avg_val_loss = validate(model, val_loader, config)

        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:3d}/{config.EPOCHS} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f} | "
              f"LR: {current_lr:.2e}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch

            save_path = os.path.join(
                config.SAVE_DIR, f"best_model_{timestamp}.pth")
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_val_loss,
                'config': vars(config)
            }, save_path)

        # 每10个epoch保存检查点
        if epoch % 10 == 0:
            save_path = os.path.join(
                config.SAVE_DIR, f"checkpoint_epoch{epoch}_{timestamp}.pth")
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_val_loss,
                'config': vars(config)
            }, save_path)

    # 训练完成
    print("\n" + "=" * 60)
    print(f"训练完成! 最佳验证损失: {best_val_loss:.6f} (Epoch {best_epoch})")
    print(f"模型保存在: {config.SAVE_DIR}")

    # 测试
    print("\n在测试集上评估...")
    best_checkpoint = os.path.join(
        config.SAVE_DIR, f"best_model_{timestamp}.pth")
    if os.path.exists(best_checkpoint):
        checkpoint = torch.load(best_checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        test_loss = validate(model, test_loader, config)
        print(f"测试集损失: {test_loss:.6f}")

    return model


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    train()
