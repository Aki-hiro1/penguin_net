import torch
import numpy as np
from pathlib import Path
import sys
import rasterio
from tqdm import tqdm
import time

src_dir = Path(__file__).parent.parent
sys.path.append(str(src_dir))

try:
    from tools.file_tool import get_tif_path, load_image_by_projection
    from models.unet_model import UNet
    from models.se_unet_model import SE_UNet
    from tools.calculate_tool import indices_generate
except ImportError as e:
    print(e)


def convert_6ch_to_19ch(data_6ch: np.ndarray) -> np.ndarray:
    """
    将6通道影像转换为19通道（6原始 + 13指数）

    Args:
        data_6ch: [6, H, W] 原始影像

    Returns:
        data_19ch: [19, H, W] 拼接后的数据
    """
    _, H, W = data_6ch.shape

    # 展平为 [6, H*W]
    data_6ch_flat = data_6ch.reshape(6, -1)  # [6, H*W]

    # 计算指数（indices_generate 期望输入 [6, N]）
    indices = indices_generate(data_6ch_flat)  # [13, H*W]

    # 拼接 6 + 13 = 19 通道
    data_19ch_flat = np.concatenate(
        [data_6ch_flat, indices], axis=0)  # [19, H*W]

    # 重塑为 [19, H, W]
    data_19ch = data_19ch_flat.reshape(19, H, W)

    return data_19ch.astype(np.float32)


def predict_tiled(
    model,
    image_data: np.ndarray,
    tile_size: int = 512,
    batch_size: int = 16,
    device: torch.device = None
) -> np.ndarray:
    """
    分块预测整张影像

    参数:
        model: 训练好的模型
        image_data: [19, H, W] 影像数据（6原始 + 13指数）
        tile_size: 分块大小
        batch_size: 每批处理多少个 tile
        device: 设备

    返回:
        class_map: [H, W] 类别图 (0=背景, 1=目标, 2=空类)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    _, H, W = image_data.shape

    # 输出数组
    class_map = np.zeros((H, W), dtype=np.uint8)

    # 计算需要多少个 tile
    tiles_h = (H + tile_size - 1) // tile_size
    tiles_w = (W + tile_size - 1) // tile_size
    print(f"影像尺寸: {H}×{W}, 分块: {tiles_h}×{tiles_w} = {tiles_h * tiles_w} 块")

    # 收集所有 tile
    tile_list = []
    tile_positions = []

    for i in range(tiles_h):
        for j in range(tiles_w):
            row_start = i * tile_size
            row_end = min(row_start + tile_size, H)
            col_start = j * tile_size
            col_end = min(col_start + tile_size, W)

            tile = image_data[:, row_start:row_end, col_start:col_end]
            tile_list.append(tile)
            tile_positions.append((row_start, row_end, col_start, col_end))

    # 分批预测
    with torch.no_grad():
        for idx in tqdm(range(0, len(tile_list), batch_size), desc="预测中"):
            batch_tiles = tile_list[idx:idx + batch_size]
            batch_positions = tile_positions[idx:idx + batch_size]

            # 转为 Tensor
            batch_tensor = torch.from_numpy(
                # [B, 19, tile_H, tile_W]
                np.stack(batch_tiles, axis=0)).to(device)

            # 模型预测（注意：你的模型需要 img 和 indices 分开输入）
            # 这里使用 19 通道直接输入，需要调整模型或拆分
            # 拆分 19 通道为 6 + 13
            img_batch = batch_tensor[:, :6, :, :]      # [B, 6, H, W]
            indices_batch = batch_tensor[:, 6:, :, :]  # [B, 13, H, W]

            outputs = model(img_batch, indices_batch)   # [B, 3, H, W]

            # 取 argmax 得到类别
            preds = outputs.argmax(dim=1).cpu().numpy()  # [B, tile_H, tile_W]

            # 填回原图
            for pred, (row_start, row_end, col_start, col_end) in zip(preds, batch_positions):
                class_map[row_start:row_end, col_start:col_end] = pred

    return class_map


def main():
    # ============ 配置 ============
    checkpoint_path = Path(
        r".\checkpoints\best_model_20260920_114239.pth")  # 你的最佳模型
    # checkpoint_path = Path(r".\src\models\unet_fold_more_ch.pth")

    # 输入影像文件夹（包含6个波段TIF）
    tif_folder = Path(r"d:\outputimg\2023_21")

    # 也可以指定中心点坐标裁剪（参考你的原代码）
    center_x = 511720
    center_y = 1715640
    crop_width = 512
    crop_height = 512

    tile_size = 512      # 推理时的分块大小
    batch_size = 16      # 每批处理多少块

    output_path = Path(r".\output\prediction_result.tif")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 模型参数（与训练时一致）
    IN_CHANNELS_A = 6
    IN_CHANNELS_B = 13
    NUM_CLASSES = 3
    FEATURES = [64, 128, 256, 512]
    SE_REDUCTION = 16

    # ============ 设备 ============
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ============ 加载数据 ============
    print("加载影像...")
    tif_paths = get_tif_path(tif_folder, satellite_index=1)

    # 如果指定了中心点，则裁剪
    if center_x is not None and center_y is not None and crop_width is not None and crop_height is not None:
        image_data, profile, crs, shape = load_image_by_projection(
            tif_paths, center_x, center_y, crop_width, crop_height
        )
    else:
        # 加载整张影像
        image_data, profile, crs, shape = load_image_by_projection(tif_paths)

    # image_data: [6, H, W]
    print(f"影像尺寸: {image_data.shape}")

    print("转换 6→19 通道...")
    image_data = convert_6ch_to_19ch(image_data)
    print(f"影像尺寸 (19通道): {image_data.shape}")

    # ============ 加载模型 ============
    print("加载模型...")
    model = SE_UNet(
        in_channels_a=IN_CHANNELS_A,
        in_channels_b=IN_CHANNELS_B,
        num_classes=NUM_CLASSES,
        features=FEATURES,
        se_reduction=SE_REDUCTION
    ).to(device)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ 模型加载成功: {checkpoint_path}")
    print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"   Best Loss: {checkpoint.get('best_loss', 'unknown')}")

    # ============ 预测 ============
    print("开始预测...")
    start_time = time.time()
    class_map = predict_tiled(
        model=model,
        image_data=image_data,
        tile_size=tile_size,
        batch_size=batch_size,
        device=device
    )
    elapsed = time.time() - start_time
    print(f"⏱️ 预测耗时: {elapsed:.2f} 秒")

    # ============ 保存结果 ============
    print("保存结果...")
    out_profile = profile.copy()
    out_profile.update({
        'count': 1,
        'dtype': 'uint8',
        'nodata': 255,
        'compress': 'deflate'
    })

    with rasterio.open(output_path, 'w', **out_profile) as dst:
        dst.write(class_map, 1)

    print(f"✅ 预测完成! 结果保存在: {output_path}")

    # 打印类别统计
    unique, counts = np.unique(class_map, return_counts=True)
    print("\n📊 类别分布:")
    class_names = ['背景', '目标', '空类']
    for cls, count in zip(unique, counts):
        name = class_names[cls] if cls < len(class_names) else f"类别{cls}"
        print(f"  {name}: {count} 像素 ({count / class_map.size * 100:.2f}%)")


if __name__ == "__main__":
    main()
