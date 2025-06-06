# me  –  Script TOP DAT (full, standalone, your variable names)
import math, cv2, numpy as np

# ========================參數=========================
prams_visual_debug = False
human_circle_fill = True

# ───────────────────  HELPERS  ──────────────────────────────────

def flip_image_both_axes(img: np.ndarray) -> np.ndarray:
    """
    對一張 BGR 圖像進行水平與垂直翻轉（不含 alpha 通道）

    Parameters:
        img (np.ndarray): 3 通道 BGR 格式的 OpenCV 圖像

    Returns:
        np.ndarray: 翻轉後的圖像
    """
    if img is None:
        raise ValueError("輸入圖像為 None")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("輸入圖像需為 BGR 格式（三通道）")

    # flipCode = -1 表示水平 + 垂直翻轉
    return cv2.flip(img, -1)
def process_image_white_BG_2_black_BG(img: np.ndarray) -> np.ndarray:
    """
    處理不含 alpha 的 BGR 圖像：
    - 將非黑非白像素設為白色
    - 黑色變白色，白色變黑色

    Parameters:
        img (np.ndarray): 3 通道 BGR 圖像

    Returns:
        np.ndarray: 處理後的圖像
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("輸入圖像需為 BGR 格式（三通道）")

    bgr = img.copy()

    # 建立黑與白的遮罩
    black_mask = np.all(bgr == [0, 0, 0], axis=-1)
    white_mask = np.all(bgr == [255, 255, 255], axis=-1)
    other_mask = ~(black_mask | white_mask)

    # 將非黑非白像素設為白色
    bgr[other_mask] = [255, 255, 255]

    # 更新遮罩
    black_mask = np.all(bgr == [0, 0, 0], axis=-1)
    white_mask = np.all(bgr == [255, 255, 255], axis=-1)

    # 黑白反轉
    bgr[black_mask] = [255, 255, 255]
    bgr[white_mask] = [0, 0, 0]

    return bgr



def group_and_draw_circles(img: np.ndarray, x_pct: float, y_pct: float, r: int) -> np.ndarray:
    h, w  = img.shape[:2]
    dx, dy = int(w*x_pct/100), int(h*y_pct/100)
    mask   = np.any(img != 255, axis=2)
    mask_crop = mask[dy:h-dy, dx:w-dx]
    mask_full = np.zeros_like(mask); mask_full[dy:h-dy, dx:w-dx] = mask_crop
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
    _, labels = cv2.connectedComponents(cv2.dilate(mask_full.astype(np.uint8), kernel))
    out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for lab in range(1, labels.max()+1):
        ys, xs = np.where((labels == lab) & mask_full)
        if xs.size:
            fill_config = 2 # default circle edge px
            if human_circle_fill:
                fill_config = -1
            else:
                fill_config = 2
            cv2.circle(out, (int(xs.mean()), int(ys.mean())), r, _get_color_bgr('black'),fill_config)#(0, 0, 255), 2)
    #return out
    out_and_flip=flip_image_both_axes(out)
    if prams_visual_debug:
        return out_and_flip
    else: 
        return process_image_white_BG_2_black_BG(out_and_flip)

#def _draw_frame_on_ax(ax, radii_frame, angles_frame, sensor_trans, colors_sensor):
def _process_sensor_data(radii_frame, angles_frame, sensor_trans):
    sensor_coords = []
    for s in range(4):
        if not radii_frame[s]:
            continue
        tx, ty = sensor_trans[s]
        ptsx, ptsy = [], []
        for r, ang_deg in zip(radii_frame[s], angles_frame[s]):
            if r > 15.0:
                continue
            a = math.radians(ang_deg)
            ptsx.append(r * math.sin(a) + tx)
            ptsy.append(r * math.cos(a) + ty)
        #if ptsx:
        #    ax.scatter(ptsx, ptsy, s=1, color=colors_sensor[s], marker='.')
        sensor_coords.append((ptsx, ptsy))
    return sensor_coords

# 新的
def _get_color_bgr(color_name):
    """
    Convert color name to BGR tuple for OpenCV.
    """
    color_map = {
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'purple': (128, 0, 128),
        'black': (0, 0, 0),
        'white': (255, 255, 255)
    }
    return color_map.get(color_name, (0, 0, 0))  # Default to black
# 新的
def _world_to_pixel(x_world, y_world, canvas_w_px, canvas_h_px, plot_x_half, plot_y_half):
    """
    Convert world coordinates to pixel coordinates.
    """
    # Normalize world coordinates to [0, 1]
    x_norm = (x_world + plot_x_half) / (2 * plot_x_half)
    y_norm = (y_world + plot_y_half) / (2 * plot_y_half)
    
    # Convert to pixel coordinates (note: y is flipped for image coordinates)
    x_pixel = int(x_norm * canvas_w_px)
    y_pixel = int((1 - y_norm) * canvas_h_px)  # Flip y-axis
    
    # Clamp to valid pixel range
    x_pixel = max(0, min(canvas_w_px - 1, x_pixel))
    y_pixel = max(0, min(canvas_h_px - 1, y_pixel))
    
    return x_pixel, y_pixel

def frame2opencvIMG(frame_radii_data, frame_angles_data,
                    canvas_w_px, canvas_h_px,
                    plot_x_half, plot_y_half,
                    sensor_trans, colors_sensor,
                    fixed_dpi):

    # 新的
    image = np.full((canvas_h_px, canvas_w_px, 3), 255, dtype=np.uint8)
    
    # Process sensor data to get coordinates
    sensor_coords = _process_sensor_data(frame_radii_data, frame_angles_data, sensor_trans)
    
    # Draw points for each sensor
    for sensor_idx, (x_coords, y_coords) in enumerate(sensor_coords):
        if x_coords and y_coords:
            color_bgr = _get_color_bgr(colors_sensor[sensor_idx])
            
            for x_world, y_world in zip(x_coords, y_coords):
                x_pixel, y_pixel = _world_to_pixel(x_world, y_world, canvas_w_px, canvas_h_px, plot_x_half, plot_y_half)
                # Draw a small circle for each point (radius=1 for small dots)
                cv2.circle(image, (x_pixel, y_pixel), 1, color_bgr, -1)

    return group_and_draw_circles(image, 5.0, 5.0, 55)
    # ==============這裡是參數，很重要，55那個值如果調成20會顯示雙腳，55會是人的肚子的腰圍=========================
    # 5.0、5.0 是 邊界 margin 的百分比

# ───────────────────  PARAM STUB  ───────────────────────────────
def onSetupParameters(scriptOp):
    # 無自訂參數
    return

# ───────────────────  MAIN COOK  ────────────────────────────────
def onCook(scriptOp):
    
    chop = op('script_chop1')
    r_vals_1 = chop['radius1'].vals  # numpy array	
    r_vals_2 = chop['radius2'].vals
    r_vals_3 = chop['radius3'].vals
    r_vals_4 = chop['radius4'].vals
    a_vals_1 = chop['angle1'].vals
    a_vals_2  = chop['angle2'].vals
    a_vals_3  = chop['angle3'].vals
    a_vals_4  = chop['angle4'].vals

    # 2️⃣  把四組資料組成函式需要的 list
    radii_frame  = [r_vals_1, r_vals_2, r_vals_3, r_vals_4]
    angles_frame = [a_vals_1, a_vals_2, a_vals_3, a_vals_4]

    # 3️⃣  固定參數
    W, H   = 1920,1080#1280, 720
    DPI    = 100
    plot_x_half = 6.7
    plot_y_half = 6.7 * H / W
    sensor_trans = [(-6.7, -1.7), (6.7, 1.0),
                    (6.7, -1.7), (-6.7, 1.0)]
    colors_sensor = ['red', 'green', 'blue', 'purple']

    # 4️⃣  產生影像→傳給 Script TOP
    img_bgr = frame2opencvIMG(radii_frame, angles_frame,
                              W, H, plot_x_half, plot_y_half,
                              sensor_trans, colors_sensor, DPI)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    scriptOp.copyNumpyArray(img_rgb)
    return
