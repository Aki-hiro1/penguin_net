import torch
import numpy as np
from pathlib import Path
import sys
import rasterio
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import geopandas as gpd
from rasterio.features import geometry_mask
import time

src_dir = Path(__file__).parent.parent
sys.path.append(str(src_dir))

try:
    from tools.file_tool import get_tif_path, load_image_by_projection
    from models.unet_model import UNet
    from models.se_unet_model import SE_UNet
    from tools.calculate_tool import indices_generate
except ImportError as e:
    print(f"❌ 导入失败: {e}")


def convert_6ch_to_19ch(data_6ch: np.ndarray) -> np.ndarray:
    _, H, W = data_6ch.shape
    data_6ch_flat = data_6ch.reshape(6, -1)
    indices = indices_generate(data_6ch_flat)
    data_19ch_flat = np.concatenate([data_6ch_flat, indices], axis=0)
    data_19ch = data_19ch_flat.reshape(19, H, W)
    return data_19ch.astype(np.float32)


def predict_image(model, image_data_19ch, device, tile_size=512):
    model.eval()
    _, H, W = image_data_19ch.shape

    if H <= tile_size and W <= tile_size:
        img = image_data_19ch[:6, :, :]
        indices = image_data_19ch[6:, :, :]
        with torch.no_grad():
            img_tensor = torch.from_numpy(img[None]).to(device)
            indices_tensor = torch.from_numpy(indices[None]).to(device)
            output = model(img_tensor, indices_tensor)
            pred = output.argmax(dim=1).cpu().numpy()[0]
        return pred

    class_map = np.zeros((H, W), dtype=np.uint8)
    for row in range(0, H, tile_size):
        for col in range(0, W, tile_size):
            row_end = min(row + tile_size, H)
            col_end = min(col + tile_size, W)
            tile = image_data_19ch[:, row:row_end, col:col_end]
            img_tile = tile[:6, :, :]
            indices_tile = tile[6:, :, :]
            with torch.no_grad():
                img_tensor = torch.from_numpy(img_tile[None]).to(device)
                indices_tensor = torch.from_numpy(
                    indices_tile[None]).to(device)
                output = model(img_tensor, indices_tensor)
                pred = output.argmax(dim=1).cpu().numpy()[0]
            class_map[row:row_end, col:col_end] = pred
    return class_map


def get_shp_mask(shp_path, crs, shape, transform):
    gdf = gpd.read_file(shp_path)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    mask = geometry_mask(
        geometries=gdf.geometry,
        out_shape=shape,
        transform=transform,
        invert=True
    )
    return mask


def normalize_band(band):
    band = band.astype(np.float32)
    if band.max() > 1.5:
        band = band / 10000.0
    p2 = np.percentile(band, 2)
    p98 = np.percentile(band, 98)
    band = np.clip((band - p2) / (p98 - p2 + 1e-8), 0, 1)
    return band


def main():
    # ============ 配置 ============
    checkpoint_path = Path(r".\checkpoints\best_model_20260920_114239.pth")
    tif_folder = Path(r"d:\outputimg\2023_20")

    center_x = 511720
    center_y = 1715640
    crop_size = 500

    rock_shp_path = Path(
        r"C:\Users\39632\Desktop\dachuang\add_rockoutcrop_landsat_v7.3\add_rockoutcrop_landsat_v7.3.shp"
    )
    # rock_shp_path = None

    output_path = Path(r".\output\visualization.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
    datas, profile, crs, shape = load_image_by_projection(
        tif_paths, center_x, center_y, crop_size, crop_size
    )
    print(f"影像尺寸: {datas.shape}")

    # ============ 岩石掩膜 ============
    rock_mask = None
    if rock_shp_path is not None and rock_shp_path.exists():
        print("生成岩石掩膜...")
        rock_mask = get_shp_mask(
            rock_shp_path, crs, shape, profile['transform'])
        print(f"岩石掩膜: {rock_mask.sum()} 个像素")

    # ============ 6→19通道 ============
    print("转换 6→19 通道...")
    datas_19ch = convert_6ch_to_19ch(datas)
    print(f"19通道尺寸: {datas_19ch.shape}")

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

    # ============ 预测 ============
    print("预测中...")
    start_time = time.time()
    class_map = predict_image(model, datas_19ch, device, tile_size=512)
    print(f"⏱️ 预测耗时: {time.time() - start_time:.2f} 秒")

    unique, counts = np.unique(class_map, return_counts=True)
    print(f"类别分布: {dict(zip(unique, counts))}")

    # ============ 可视化 ============
    print("生成可视化...")

    # 原图
    nir = normalize_band(datas[3])
    green = normalize_band(datas[1])
    blue = normalize_band(datas[0])
    rgb_image = np.stack([nir, green, blue], axis=-1)

    # ============================================================
    # 分类叠加（用于右图）：红色=目标，蓝色=背景，绿色=空类
    # ============================================================
    overlay_right = rgb_image.copy()

    mask_target = (class_map == 0)
    overlay_right[mask_target, 0] = 1.0
    overlay_right[mask_target, 1] = 0.0
    overlay_right[mask_target, 2] = 0.0

    mask_background = (class_map == 1)
    overlay_right[mask_background, 0] = 0.0
    overlay_right[mask_background, 1] = 0.0
    overlay_right[mask_background, 2] = 1.0

    mask_empty = (class_map == 2)
    overlay_right[mask_empty, 0] = 0.0
    overlay_right[mask_empty, 1] = 1.0
    overlay_right[mask_empty, 2] = 0.0

    # ============================================================
    # 中间图（sub1）：只显示目标（红色）和空类（绿色），背景（类别1）透明
    # ============================================================
    overlay_mid = rgb_image.copy()

    # 类别0：目标 → 红色（覆盖原图）
    overlay_mid[mask_target, 0] = 1.0
    overlay_mid[mask_target, 1] = 0.0
    overlay_mid[mask_target, 2] = 0.0

    # 类别2：空类 → 绿色（覆盖原图）
    overlay_mid[mask_empty, 0] = 0.0
    overlay_mid[mask_empty, 1] = 1.0
    overlay_mid[mask_empty, 2] = 0.0

    # 类别1：背景 → 保持原图显示（即"透明"效果）
    # ⬅️ 不操作，保持原图的RGB值
    # ============================================================

    # ============================================================
    # 右图：分类 + 岩石掩膜
    # ============================================================
    overlay_rock = overlay_right.copy()
    if rock_mask is not None:
        rock_alpha = 0.5
        overlay_rock[rock_mask, 0] = overlay_rock[rock_mask, 0] * \
            (1 - rock_alpha) + 0.0 * rock_alpha
        overlay_rock[rock_mask, 1] = overlay_rock[rock_mask, 1] * \
            (1 - rock_alpha) + 1.0 * rock_alpha
        overlay_rock[rock_mask, 2] = overlay_rock[rock_mask, 2] * \
            (1 - rock_alpha) + 1.0 * rock_alpha

    # ============ 绘图 ============
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # 左图：原图
    axes[0].imshow(rgb_image)
    axes[0].set_title('Original Image (NIR, Green, Blue)')
    axes[0].axis('off')

    # 中图：目标（红）+ 空类（绿），背景透明（显示原图）
    axes[1].imshow(overlay_mid)
    axes[1].set_title('Target (Red) + Unknown (Green), Background Transparent')
    axes[1].axis('off')
    legend_elements1 = [
        Patch(facecolor=[1, 0, 0], edgecolor='white',
              label='Target (Class 0)'),
        Patch(facecolor=[0, 1, 0], edgecolor='white',
              label='Unknown (Class 2)'),
        Patch(facecolor='none', edgecolor='white',
              label='Background (Class 1, Transparent)'),
    ]
    axes[1].legend(handles=legend_elements1,
                   bbox_to_anchor=(1.05, 1), loc='upper left')

    # 右图：分类 + 岩石掩膜
    axes[2].imshow(overlay_rock)
    if rock_mask is not None:
        axes[2].set_title('Classification + Rock Mask (Cyan = Rock)')
        legend_elements2 = [
            Patch(facecolor=[1, 0, 0], edgecolor='white',
                  label='Target (Class 0)'),
            Patch(facecolor=[0, 0, 1], edgecolor='white',
                  label='Background (Class 1)'),
            Patch(facecolor=[0, 1, 0], edgecolor='white',
                  label='Unknown (Class 2)'),
            Patch(facecolor=[0, 1, 1], edgecolor='white',
                  alpha=0.5, label='Rock Mask'),
        ]
    else:
        axes[2].set_title('Classification Result')
        legend_elements2 = [
            Patch(facecolor=[1, 0, 0], edgecolor='white',
                  label='Target (Class 0)'),
            Patch(facecolor=[0, 0, 1], edgecolor='white',
                  label='Background (Class 1)'),
            Patch(facecolor=[0, 1, 0], edgecolor='white',
                  label='Unknown (Class 2)'),
        ]
    axes[2].axis('off')
    axes[2].legend(handles=legend_elements2,
                   bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图像已保存: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
