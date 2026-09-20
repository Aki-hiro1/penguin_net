import rasterio
import yaml
import numpy as np
from pathlib import Path
import geopandas as gpd
from rasterio.transform import rowcol

MIN_POINTS_PER_PATCH = 20
CLASS_MAPPING = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1}


def get_tif_path(folder_path, satellite_index=0):
    if satellite_index == 0:
        BAND_NAME = np.array(['B2', 'B3', 'B4', 'B5', 'B6', 'B7'])
    elif satellite_index == 1:
        BAND_NAME = np.array(["B2", "B3", "B4", "B8", "B11", "B12"])
    else:
        return [None]*6
    BAND_INDEX = {name: i for i, name in enumerate(BAND_NAME)}

    folder_path = Path(folder_path)
    image_paths = [None]*6
    tif_paths = [path for path in folder_path.iterdir() if path.is_file()
                 and path.suffix.lower() == '.tif']

    for path in tif_paths:
        band_name = path.stem.split('_')[-1]
        if satellite_index == 1:
            band_name = path.stem.split('_')[0]
        if band_name in BAND_NAME:
            image_paths[BAND_INDEX[band_name]] = path
    return image_paths


def load_image_by_projection(image_paths, center_x=None, center_y=None, width=None, height=None):
    """
    根据投影坐标加载图像（适用于投影坐标系TIF）

    :param image_paths: 图像路径列表
    :param center_x: 中心点X坐标（米）
    :param center_y: 中心点Y坐标（米）
    :param width: 裁剪宽度（像素）
    :param height: 裁剪高度（像素）
    :return: datas, profile, crs, shape
    """
    datas = []

    with rasterio.open(image_paths[0]) as src:
        if src.crs.is_geographic:
            raise ValueError(f"TIF不是投影坐标系，当前坐标系: {src.crs}")

        crs = src.crs
        transform = src.transform
        profile = src.profile.copy()

        if center_x is None or center_y is None or width is None or height is None:
            window = None
            shape = (src.height, src.width)
        else:
            row, col = src.index(center_x, center_y)

            col_start = max(0, col - width // 2)
            row_start = max(0, row - height // 2)
            col_end = min(src.width, col_start + width)
            row_end = min(src.height, row_start + height)

            if col_end - col_start < width:
                col_start = max(0, col_end - width)
            if row_end - row_start < height:
                row_start = max(0, row_end - height)

            window = rasterio.windows.Window(
                col_start, row_start, col_end - col_start, row_end - row_start)
            transform = rasterio.windows.transform(window, src.transform)
            profile.update(
                {'width': window.width, 'height': window.height, 'transform': transform})
            shape = (window.height, window.width)

    for path in image_paths:
        with rasterio.open(path) as src:
            datas.append(src.read(1, window=window))

    datas = np.stack(datas, axis=0)
    datas = datas.astype(np.float32) / 10000
    return datas, profile, crs, shape


def map_labels(labels: np.ndarray, mapping: dict) -> np.ndarray:
    """将原始类别映射为目标类别"""
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label not in mapping:
            raise KeyError(f"标签 {label} 不在映射字典中。可用: {list(mapping.keys())}")

    mapped = np.vectorize(mapping.get)(labels)
    return mapped.astype(np.int64)


def read_shp(shp_path):
    shp_path = Path(shp_path)
    if not shp_path.exists():
        raise FileNotFoundError(f"Shapefile 不存在: {shp_path}")

    gdf = gpd.read_file(shp_path)

    if not gdf.geometry.geom_type.isin(["Point", "MultiPoint"]).all():
        raise ValueError(
            f"Shapefile 必须包含 Point 几何，当前类型: {gdf.geometry.geom_type.unique()}")

    return gdf


def sample_by_class(rows, cols, raw_labels, sample_num: list) -> tuple:
    """
    按原始类别采样点，每类取 sample_num[i] 个

    参数:
        rows: 行号数组 [N]
        cols: 列号数组 [N]
        raw_labels: 原始类别数组 [N] (0-7)
        sample_num: 每类采样数量列表，如 [300, 200, 200, 200, 100, 300, 60, 180]

    返回:
        采样后的 rows, cols, raw_labels
    """
    if sample_num is None:
        return rows, cols, raw_labels

    selected_rows = []
    selected_cols = []
    selected_labels = []

    unique_classes = np.unique(raw_labels)

    for cls in unique_classes:
        # 找到该类别的所有点
        mask = raw_labels == cls
        cls_rows = rows[mask]
        cls_cols = cols[mask]
        cls_labels = raw_labels[mask]

        n_available = len(cls_rows)
        n_target = sample_num[cls] if cls < len(sample_num) else n_available

        # 如果目标数量大于可用数量，全部使用
        if n_target >= n_available:
            selected_rows.extend(cls_rows)
            selected_cols.extend(cls_cols)
            selected_labels.extend(cls_labels)
            print(f"    类别 {cls}: 可用 {n_available}，全部使用")
        else:
            # 随机采样
            indices = np.random.choice(
                n_available, size=n_target, replace=False)
            selected_rows.extend(cls_rows[indices])
            selected_cols.extend(cls_cols[indices])
            selected_labels.extend(cls_labels[indices])
            print(f"    类别 {cls}: 可用 {n_available}，采样 {n_target}")

    return np.array(selected_rows), np.array(selected_cols), np.array(selected_labels)


def points_to_pixels(
        gdf,
        transform,
        image_height,
        image_width,
        label_field="Id",
        label_mapping=None,
        return_fields=None,
        sample_num=None):  # ← 新增参数
    """
    将点 shapefile 的坐标转换为像素行列号

    参数:
        gdf: GeoDataFrame
        transform: 影像的仿射变换
        image_height: 影像高度
        image_width: 影像宽度
        label_field: 类别字段名（默认 "Id"）
        label_mapping: 类别映射字典
        return_fields: 要返回的字段列表
        sample_num: 每类采样数量列表，如 [300, 200, 200, 200, 100, 300, 60, 180]

    返回:
        rows: 行号数组 [N]
        cols: 列号数组 [N]
        labels: 映射后的类别标签数组 [N]
        field_values: 字段值字典
    """
    # 获取点的 X、Y 坐标
    x_coords = gdf.geometry.x.to_numpy()
    y_coords = gdf.geometry.y.to_numpy()

    # 地理坐标 → 像素行列
    rows, cols = rowcol(transform, x_coords, y_coords)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)

    # 检查哪些点落在影像范围内
    in_bounds = (rows >= 0) & (rows < image_height) & (
        cols >= 0) & (cols < image_width)

    rows_valid = rows[in_bounds]
    cols_valid = cols[in_bounds]

    # 提取类别标签
    if label_field not in gdf.columns:
        raise KeyError(f"找不到字段 '{label_field}'。可用字段: {gdf.columns.tolist()}")

    raw_labels = gdf[label_field].to_numpy()[in_bounds].astype(np.int64)

    dropped = int((~in_bounds).sum())
    if dropped > 0:
        print(f"⚠️ 丢弃了 {dropped} 个落在影像外的点")

    if len(rows_valid) == 0:
        raise ValueError("没有点落在影像范围内")

    # ============================================================
    # 🔽 按原始类别采样
    # ============================================================
    print(f"  采样前: {len(rows_valid)} 个点")
    if sample_num is not None:
        rows_valid, cols_valid, raw_labels = sample_by_class(
            rows_valid, cols_valid, raw_labels, sample_num
        )
    print(f"  采样后: {len(rows_valid)} 个点")
    # ============================================================

    # ============================================================
    # 🔽 类别映射（8类 → 2类）
    # ============================================================
    if label_mapping is not None:
        labels = map_labels(raw_labels, label_mapping)
    else:
        labels = raw_labels
    # ============================================================

    # 提取指定字段的值
    field_values = {}
    if return_fields:
        for field_name in return_fields:
            if field_name not in gdf.columns:
                raise KeyError(
                    f"找不到字段 '{field_name}'。可用字段: {gdf.columns.tolist()}")
            field_values[field_name] = gdf[field_name].to_numpy()[in_bounds]

    return rows_valid, cols_valid, labels, field_values


def load_scenes(manifest_path: str | Path, patch_size: int = 128, sample_num: list = None) -> list[dict]:
    """
    从 manifest 文件读取所有场景，以每个样本点为中心裁切图幅

    过滤规则：
        - 中心点为类别0：图幅内样本点总数 >= MIN_POINTS_FOR_CLASS0 才保留
        - 中心点为类别1：不过滤，全部保留

    参数:
        manifest_path: context_manifest.yaml 路径
        patch_size: 图幅尺寸（默认128）
        sample_num: 每类采样数量列表，如 [300, 200, 200, 200, 100, 300, 60, 180]

    返回:
        图幅列表，每个图幅包含:
            - image_data: [6, patch_size, patch_size]
            - points: {'rows': [], 'cols': [], 'labels': []}
            - scene_id: 来源场景名
            - center_row, center_col: 中心点像素坐标（原图）
            - center_label: 中心点类别（0/1）
            - n_points: 图幅内总样本点数
    """
    manifest_path = Path(manifest_path)
    base_dir = manifest_path.parent
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = yaml.safe_load(f)

    patches = []
    bands = ["B2", "B3", "B4", "B8", "B11", "B12"]

    # 默认采样数量
    if sample_num is None:
        sample_num = [300, 200, 200, 200, 100, 300, 60, 180]

    for entry in manifest['scenes']:
        scene_id = entry['id']
        image_dir = Path(base_dir / entry['image_dir'])
        label_path = Path(base_dir / entry['label_path'])

        print(f"\n加载场景: {scene_id}")

        # 1. 加载6通道整张影像
        tif_paths = get_tif_path(image_dir, satellite_index=1)
        tif_data, tif_profile, crs, shape = load_image_by_projection(tif_paths)
        H, W = shape
        print(f"  影像尺寸: {H}×{W}")

        # 2. 加载点SHP
        gdf = read_shp(label_path)
        if gdf.crs is not None and tif_profile.get('crs') is not None:
            if gdf.crs != tif_profile['crs']:
                print(f"  ⚠️ 重投影: {gdf.crs} → {tif_profile['crs']}")
                gdf = gdf.to_crs(tif_profile['crs'])

        # 3. 所有点转为像素坐标 + 类别映射（传入 sample_num）
        rows, cols, labels, infos = points_to_pixels(
            gdf,
            tif_profile['transform'],
            tif_profile['height'],
            tif_profile['width'],
            label_field="Id",
            label_mapping=CLASS_MAPPING,
            return_fields=bands,
            sample_num=sample_num  # ⬅️ 传入采样数量
        )

        print(f"  总有效点: {len(rows)}")

        # 4. 统计类别分布
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  类别分布: {dict(zip(unique, counts))}")

        # 5. 对每个点，裁切图幅
        half = patch_size // 2
        patch_count = 0
        filtered_count = 0

        for i in range(len(rows)):
            center_r = int(rows[i])
            center_c = int(cols[i])
            center_label = int(labels[i])

            # 计算裁切边界
            r_start = max(0, center_r - half)
            r_end = min(H, center_r + half)
            c_start = max(0, center_c - half)
            c_end = min(W, center_c + half)

            # 如果边界不足 patch_size，跳过
            if r_end - r_start < patch_size or c_end - c_start < patch_size:
                continue

            # 裁切影像 [6, patch_size, patch_size]
            patch_data = tif_data[:, r_start:r_end, c_start:c_end]

            # 找出该图幅内的所有点（相对坐标）
            patch_rows = []
            patch_cols = []
            patch_labels = []

            for j in range(len(rows)):
                r = int(rows[j])
                c = int(cols[j])
                if r_start <= r < r_end and c_start <= c < c_end:
                    patch_rows.append(r - r_start)
                    patch_cols.append(c - c_start)
                    patch_labels.append(labels[j])

            n_points = len(patch_rows)

            # 过滤规则：中心点为类别0时，检查图幅内样本点数量
            if center_label == 0 and n_points < MIN_POINTS_PER_PATCH:
                filtered_count += 1
                continue

            patches.append({
                'image_data': patch_data,           # [6, 128, 128]
                'points': {
                    'rows': np.array(patch_rows),   # 相对坐标
                    'cols': np.array(patch_cols),
                    'labels': np.array(patch_labels)
                },
                'scene_id': scene_id,
                'center_row': center_r,
                'center_col': center_c,
                'center_label': center_label,
                'n_points': n_points
            })

            patch_count += 1

        print(f"  生成图幅: {patch_count} 个")
        print(f"  过滤掉(类别0点数<{MIN_POINTS_PER_PATCH}): {filtered_count} 个")

    print(f"\n总图幅数: {len(patches)}")
    return patches
