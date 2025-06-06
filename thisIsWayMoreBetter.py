# me  –  Script TOP DAT
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
    """
    1. 先建立 mask_full（與原本邏輯相同）； 
    2. 用手動畫圓的方式，把 mask_full 中每一個 True 的點，
       畫成半徑為 r 的實心圓到一張新的二值圖 dilated_manual；
    3. 對 dilated_manual 用 connectedComponentsWithStats 找出每個填充後大圓區塊
       的標籤、質心（centroids）等資訊；
    4. 最後把質心位置再畫一次半徑為 r 的圓到輸出影像上。
    """
    h, w = img.shape[:2]
    dx, dy = int(w * x_pct / 100), int(h * y_pct / 100)
    # --- 1. 計算 mask_full 的方式與你原本相同 ---
    mask   = np.any(img != 255, axis=2)
    mask_crop = mask[dy:h-dy, dx:w-dx]
    mask_full = np.zeros_like(mask)
    mask_full[dy:h-dy, dx:w-dx] = mask_crop

    # --- 2. 手動「畫大圓」到一張新的二值圖上（取代 getStructuringElement + dilate） ---
    # 建立一張全 0 的二值圖，大小同 mask_full
    dilated_manual = np.zeros_like(mask_full, dtype=np.uint8)

    # 把 mask_full 中任一 True 點，畫成半徑 r 的實心圓到 dilated_manual
    ys_nonzero, xs_nonzero = np.nonzero(mask_full)
    for (y0, x0) in zip(ys_nonzero, xs_nonzero):
        # 在 dilated_manual 上畫一個「白色實心圓」，半徑 r
        cv2.circle(dilated_manual, (x0, y0), r, 255, -1)

    # --- 3. 用 connectedComponentsWithStats 找每個「大圓群」的質心与統計資訊 ---
    # 這裡一定要先轉成 0/1 二值圖 (uint8)，connectedComponentsWithStats 會把 255 當成前景
    # connectivity=8 可以讓對角線也算是連通
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated_manual, connectivity=8)

    # --- 4. 把質心位置再畫一次半徑為 r 的圓到輸出影像上 ---
    # 先把原始影像轉成 BGR（如果原本是 RGB）
    out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 從 1 開始到 num_labels-1（跳過背景 label=0）
    for lab in range(1, num_labels):
        # centroids[lab] 會是一個 (x_center, y_center) 的浮點數座標
        cx, cy = centroids[lab]
        # 確保型別轉成整數
        cx_int, cy_int = int(round(cx)), int(round(cy))

        # 根據 prams 設定決定填滿或只畫邊框
        if human_circle_fill:
            fill_config = -1   # -1 代表實心
        else:
            fill_config = 2    # 2 代表線寬 2 px

        # 最後在 out 上畫半徑 r 的圓
        cv2.circle(out, (cx_int, cy_int), r, _get_color_bgr('black'), fill_config)

    # 最後再翻轉、以及做黑白反轉處理
    out_and_flip = flip_image_both_axes(out)
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
    # 舊的
    # fig, ax = plt.subplots(figsize=(canvas_w_px / fixed_dpi,
    #                                 canvas_h_px / fixed_dpi),
    #                        dpi=fixed_dpi)
    # fig.patch.set_facecolor('white')
    # _draw_frame_on_ax(ax, frame_radii_data, frame_angles_data,
    #                   sensor_trans, colors_sensor)
    # ax.set_xlim(-plot_x_half, plot_x_half)
    # ax.set_ylim(-plot_y_half, plot_y_half)
    # ax.axis('off')
    # plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    # fig.canvas.draw()
    # img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    # img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    # plt.close(fig)
    # return group_and_draw_circles(img, 5.0, 5.0, 20)
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

    return group_and_draw_circles(image, 5.0, 10.0, 55)
    # ==============這裡是參數，很重要，55那個值如果調成20會顯示雙腳，55會是人的肚子的腰圍=========================
    # 5.0、5.0 是 邊界 margin 的百分比

# ───────────────────  PARAM STUB  ───────────────────────────────
def onSetupParameters(scriptOp):
    # 無自訂參數
    return

# ───────────────────  MAIN COOK  ────────────────────────────────
def onCook(scriptOp):
    
    #chop = op('script_chop1')
    #r_vals_1 = chop['radius1'].vals  # numpy array	
    #r_vals_2 = chop['radius2'].vals
    #r_vals_3 = chop['radius3'].vals
    #r_vals_4 = chop['radius4'].vals
    #a_vals_1 = chop['angle1'].vals
    #a_vals_2  = chop['angle2'].vals
    #a_vals_3  = chop['angle3'].vals
    #a_vals_4  = chop['angle4'].vals
    
    # 1️⃣  逐支感測器抓資料 —— 你要的變數名稱
    chop = op('hokuyo1')
    r_vals_1 = chop['radius'].vals
    a_vals_1 = chop['angle'].vals

    chop = op('hokuyo2')
    r_vals_2 = chop['radius'].vals
    a_vals_2 = chop['angle'].vals

    chop = op('hokuyo3')
    r_vals_3 = chop['radius'].vals
    a_vals_3 = chop['angle'].vals

    chop = op('hokuyo4')
    r_vals_4 = chop['radius'].vals
    a_vals_4 = chop['angle'].vals

    # 2️⃣  把四組資料組成函式需要的 list
    radii_frame  = [r_vals_1, r_vals_2, r_vals_3, r_vals_4]
    angles_frame = [a_vals_1, a_vals_2, a_vals_3, a_vals_4]

    # 3️⃣  固定參數
    W, H   = 1920,1080#1280, 720
    DPI    = 100
    plot_x_half = 6.7
    plot_y_half = 6.7 * H / W
    #+2
    sensor_trans = [(-6.7, -1.5), (6.7, 1.2),
                    (6.7, -1.5), (-6.7, 1.2)]
    colors_sensor = ['red', 'green', 'blue', 'purple']

    # 4️⃣  產生影像→傳給 Script TOP
    img_bgr = frame2opencvIMG(radii_frame, angles_frame,
                              W, H, plot_x_half, plot_y_half,
                              sensor_trans, colors_sensor, DPI)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    scriptOp.copyNumpyArray(img_rgb)
    return
