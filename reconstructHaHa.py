import math
import cv2
import numpy as np
from typing import NamedTuple, List, Tuple, Any

import timeit

# ========================架構=========================
# 1. 參數
# 2. 初始化
# 9999. 主渲染脈衝

# ========================參數=========================
human_circle_fill: bool = True
bool_use_virtual_data: bool = True
bool_performance_measure: bool = True

# ========================初始化=========================
def onSetupParameters(scriptOp: Any) -> None:
    return

# ───────────────────  HELPERS  ──────────────────────────────────

def flip_image_both_axes(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("輸入圖像為 None")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("輸入圖像需為 BGR 格式（三通道）")
    return cv2.flip(img, -1)

def flip_coordinates_batch(points: List[Tuple[int, int]], w: int, h: int) -> List[Tuple[int, int]]:
    return [(w - x, h - y) for x, y in points]

def process_image_white_BG_2_black_BG(img: np.ndarray) -> np.ndarray:
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("輸入圖像需為 BGR 格式（三通道）")
    bgr: np.ndarray = img.copy()
    black_mask: np.ndarray = np.all(bgr == [0, 0, 0], axis=-1)
    white_mask: np.ndarray = np.all(bgr == [255, 255, 255], axis=-1)
    other_mask: np.ndarray = ~(black_mask | white_mask)
    bgr[other_mask] = [255, 255, 255]
    black_mask = np.all(bgr == [0, 0, 0], axis=-1)
    white_mask = np.all(bgr == [255, 255, 255], axis=-1)
    bgr[black_mask] = [255, 255, 255]
    bgr[white_mask] = [0, 0, 0]
    return bgr

def scale_points_to_target(
    ptls: List[Tuple[int, int]],
    canvas_width: int,
    canvas_height: int,
    target_w: int,
    target_h: int
) -> List[Tuple[int, int]]:
    if canvas_width == 0 or canvas_height == 0:
        raise ValueError("Original canvas width and height must be non-zero")
    scale_x: float = target_w / canvas_width
    scale_y: float = target_h / canvas_height
    scaled_ptls: List[Tuple[int, int]] = [(int(x * scale_x), int(y * scale_y)) for x, y in ptls]
    return scaled_ptls

class STRUCT_group_and_draw_circles(NamedTuple):
    img: np.ndarray
    r: int
    ptls: List[Tuple[int, int]]
    maskFull: np.ndarray

def filter_points(
    ptls: List[Tuple[int, int]],
    x_full: int,
    y_full: int,
    x_percent: float,
    y_percent: float
) -> List[Tuple[int, int]]:
    x_min: float = x_full * x_percent
    x_max: float = x_full * (1 - x_percent)
    y_min: float = y_full * y_percent
    y_max: float = y_full * (1 - y_percent)
    return [
        (x, y) for x, y in ptls
        if x_min <= x <= x_max and y_min <= y <= y_max
    ]

# ===================end of helpers=========================

def group_and_draw_circles_MAIN(
    ptls: List[Tuple[int, int]],
    orig_w: int,
    orig_h: int,
    img: np.ndarray,
    x_pct: float,
    y_pct: float,
    r: int
) -> np.ndarray:
    OBJ_group_and_draw_circles: STRUCT_group_and_draw_circles = downsample480andMaskSide(ptls, orig_w, orig_h, img, x_pct, y_pct, r)
    ptlsHuman: List[Tuple[int, int]] = masterSlow_group_and_draw_circles(OBJ_group_and_draw_circles, orig_w, orig_h)
    return step3drawOut(
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        r,
        orig_w,
        orig_h,
        ptlsHuman,
        True
    )

def step3drawOut(
    img: np.ndarray,
    r: int,
    w: int,
    h: int,
    ptls: List[Tuple[int, int]],
    doBlackWhiteConvert: bool
) -> np.ndarray:
    fill_config: int = -1 if human_circle_fill else 2
    for x, y in flip_coordinates_batch(ptls, w, h):
        cv2.circle(
            img,
            (x, y),
            r,
            _get_color_bgr("black"),
            fill_config,
        )
    if doBlackWhiteConvert:
        return process_image_white_BG_2_black_BG(img)
    else:
        return img

def masterSlow_group_and_draw_circles(
    OBJ_group_and_draw_circles: STRUCT_group_and_draw_circles,
    orig_w: int,
    orig_h: int
) -> List[Tuple[int, int]]:
    points_px: List[Tuple[int, int]] = []
    kernel_small: np.ndarray = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * OBJ_group_and_draw_circles.r + 1, 2 * OBJ_group_and_draw_circles.r + 1)
    )
    labels_small: np.ndarray = cv2.connectedComponents(
        cv2.dilate(OBJ_group_and_draw_circles.img, kernel_small)
    )[1]
    labels_up: np.ndarray = cv2.resize(
        labels_small.astype(np.float32),
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int32)
    labels: np.ndarray = labels_up
    for lab in range(1, labels.max() + 1):
        ys: np.ndarray
        xs: np.ndarray
        ys, xs = np.where((labels == lab) & OBJ_group_and_draw_circles.maskFull)
        if xs.size:
            points_px.append((int(xs.mean()), int(ys.mean())))
    return points_px

def downsample480andMaskSide(
    ptls: List[Tuple[int, int]],
    w: int,
    h: int,
    img: np.ndarray,
    x_pct: float,
    y_pct: float,
    r: int
) -> STRUCT_group_and_draw_circles:
    dx: int = int(w * x_pct / 100)
    dy: int = int(h * y_pct / 100)
    mask: np.ndarray = np.any(img != 255, axis=2)
    mask_crop: np.ndarray = mask[dy: h - dy, dx: w - dx]
    mask_full: np.ndarray = np.zeros_like(mask)
    mask_full[dy: h - dy, dx: w - dx] = mask_crop
    orig_h: int
    orig_w: int
    orig_h, orig_w = mask_full.shape
    target_h: int = 480
    scale: float = target_h / orig_h
    target_w: int = int(orig_w * scale)
    mask_small: np.ndarray = cv2.resize(
        mask_full.astype(np.uint8),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    )
    r_small: int = max(1, int(round(r * scale)))
    return STRUCT_group_and_draw_circles(
        img=mask_small,
        r=r_small,
        ptls=scale_points_to_target(filter_points(ptls, w, h, x_pct, y_pct), w, h, target_w, target_h),
        maskFull=mask_full
    )

def _process_sensor_data(
    radii_frame: List[np.ndarray],
    angles_frame: List[np.ndarray],
    sensor_trans: List[Tuple[float, float]]
) -> List[Tuple[List[float], List[float]]]:
    sensor_coords: List[Tuple[List[float], List[float]]] = []
    for s in range(4):
        if not radii_frame[s]:
            continue
        tx: float
        ty: float
        tx, ty = sensor_trans[s]
        ptsx: List[float] = []
        ptsy: List[float] = []
        for r, ang_deg in zip(radii_frame[s], angles_frame[s]):
            if r > 15.0:
                continue
            a: float = math.radians(ang_deg)
            ptsx.append(r * math.sin(a) + tx)
            ptsy.append(r * math.cos(a) + ty)
        sensor_coords.append((ptsx, ptsy))
    return sensor_coords

def _get_color_bgr(color_name: str) -> Tuple[int, int, int]:
    color_map: dict[str, Tuple[int, int, int]] = {
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'purple': (128, 0, 128),
        'black': (0, 0, 0),
        'white': (255, 255, 255)
    }
    return color_map.get(color_name, (0, 0, 0))

def _world_to_pixel(
    x_world: float,
    y_world: float,
    canvas_w_px: int,
    canvas_h_px: int,
    plot_x_half: float,
    plot_y_half: float
) -> Tuple[int, int]:
    x_norm: float = (x_world + plot_x_half) / (2 * plot_x_half)
    y_norm: float = (y_world + plot_y_half) / (2 * plot_y_half)
    x_pixel: int = int(x_norm * canvas_w_px)
    y_pixel: int = int((1 - y_norm) * canvas_h_px)
    x_pixel = max(0, min(canvas_w_px - 1, x_pixel))
    y_pixel = max(0, min(canvas_h_px - 1, y_pixel))
    return x_pixel, y_pixel

def frame2opencvIMG(
    frame_radii_data: List[np.ndarray],
    frame_angles_data: List[np.ndarray],
    canvas_w_px: int,
    canvas_h_px: int,
    plot_x_half: float,
    plot_y_half: float,
    sensor_trans: List[Tuple[float, float]],
    colors_sensor: List[str],
    fixed_dpi: int
) -> np.ndarray:
    image: np.ndarray = np.full((canvas_h_px, canvas_w_px, 3), 255, dtype=np.uint8)
    points_px: List[Tuple[int, int]] = []
    sensor_coords: List[Tuple[List[float], List[float]]] = _process_sensor_data(frame_radii_data, frame_angles_data, sensor_trans)
    for sensor_idx, (x_coords, y_coords) in enumerate(sensor_coords):
        if x_coords and y_coords:
            color_bgr: Tuple[int, int, int] = _get_color_bgr(colors_sensor[sensor_idx])
            for x_world, y_world in zip(x_coords, y_coords):
                x_pixel: int
                y_pixel: int
                x_pixel, y_pixel = _world_to_pixel(x_world, y_world, canvas_w_px, canvas_h_px, plot_x_half, plot_y_half)
                points_px.append((x_pixel, y_pixel))
                cv2.circle(image, (x_pixel, y_pixel), 1, color_bgr, -1)
    return group_and_draw_circles_MAIN(points_px, canvas_w_px, canvas_h_px, image, 5.0, 10.0, 55)

def onCook(scriptOp: Any) -> None:
    chop: Any = op('script_chop1') if bool_use_virtual_data else op('hokuyo1')
    if bool_use_virtual_data:
        r_vals_1: np.ndarray = chop['radius1'].vals
        r_vals_2: np.ndarray = chop['radius2'].vals
        r_vals_3: np.ndarray = chop['radius3'].vals
        r_vals_4: np.ndarray = chop['radius4'].vals
        a_vals_1: np.ndarray = chop['angle1'].vals
        a_vals_2: np.ndarray = chop['angle2'].vals
        a_vals_3: np.ndarray = chop['angle3'].vals
        a_vals_4: np.ndarray = chop['angle4'].vals
    else:
        r_vals_1: np.ndarray = chop['radius'].vals
        a_vals_1: np.ndarray = chop['angle'].vals
        chop = op('hokuyo2')
        r_vals_2: np.ndarray = chop['radius'].vals
        a_vals_2: np.ndarray = chop['angle'].vals
        chop = op('hokuyo3')
        r_vals_3: np.ndarray = chop['radius'].vals
        a_vals_3: np.ndarray = chop['angle'].vals
        chop = op('hokuyo4')
        r_vals_4: np.ndarray = chop['radius'].vals
        a_vals_4: np.ndarray = chop['angle'].vals

    radii_frame: List[np.ndarray] = [r_vals_1, r_vals_2, r_vals_3, r_vals_4]
    angles_frame: List[np.ndarray] = [a_vals_1, a_vals_2, a_vals_3, a_vals_4]
    W: int = 1920
    H: int = 1080
    DPI: int = 100
    plot_x_half: float = 6.7
    plot_y_half: float = 6.7 * H / W
    sensor_trans: List[Tuple[float, float]] = [(-6.7, -1.5), (6.7, 1.2), (6.7, -1.5), (-6.7, 1.2)]
    colors_sensor: List[str] = ['red', 'green', 'blue', 'purple']

    img_bgr: np.ndarray = frame2opencvIMG(
        radii_frame, angles_frame,
        W, H, plot_x_half, plot_y_half,
        sensor_trans, colors_sensor, DPI
    )

    if bool_performance_measure:
        execution_time: float = timeit.timeit(
            lambda: frame2opencvIMG(
                radii_frame, angles_frame,
                W, H, plot_x_half, plot_y_half,
                sensor_trans, colors_sensor, DPI
            ),
            number=1
        )
        print(f"執行時間為 {execution_time:.6f} 秒")
    img_rgb: np.ndarray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    scriptOp.copyNumpyArray(img_rgb)
    return
