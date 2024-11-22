import cv2
import numpy as np

# 全局变量
drawing = False  # 是否正在绘制新的圆
moving = False  # 是否正在移动已存在的圆
center = (0, 0)  # 当前绘制或移动圆的圆心坐标
radius = 0  # 当前绘制的圆半径
scale = 0.5  # 缩放比例
circles = []  # 存储所有圆的信息列表 (中心和半径)
selected_circle_idx = -1  # 当前选中的圆的索引


# 鼠标回调函数
def draw_circle(event, x, y, flags, param):
    global center, radius, drawing, moving, selected_circle_idx

    if event == cv2.EVENT_LBUTTONDOWN:
        # 检查鼠标是否在某个圆内（判断是否点击现有圆）
        for idx, (cx, cy, r) in enumerate(circles):
            if np.sqrt((x / scale - cx) ** 2 + (y / scale - cy) ** 2) <= r:
                # 鼠标点击位置在圆内，开始移动
                moving = True
                selected_circle_idx = idx
                break
        else:
            # 鼠标点击位置不在任何圆内，开始绘制新圆
            drawing = True
            center = (x, y)
            radius = 0

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # 如果正在绘制新圆，动态更新半径
            radius = int(np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2))
        elif moving and selected_circle_idx != -1:
            # 如果正在移动现有圆，更新该圆的圆心位置
            circles[selected_circle_idx] = (int(x / scale), int(y / scale), circles[selected_circle_idx][2])

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            # 左键松开，完成新圆绘制
            drawing = False
            # 将新圆（缩放回原图坐标）加入到圆列表中
            circles.append((int(center[0] / scale), int(center[1] / scale), int(radius / scale)))
        elif moving:
            # 左键松开，完成圆的位置调整
            moving = False
            selected_circle_idx = -1


# 创建并保存最终的掩码图像
def create_final_mask(img_shape):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)  # 创建与图像大小一致的黑色掩码
    # 遍历所有存储的圆，并在掩码上绘制
    for (cx, cy, r) in circles:
        cv2.circle(mask, (cx, cy), r, 255, -1)  # 在掩码上画白色实心圆
    return mask


# 读取图像并缩放


img = cv2.imread(r"E:\LYX_date\yanwo_cover\2_yanwo_conver_data\data/202.jpg")  # 替换成你想要操作的图像路径
la="E:\LYX_date\yanwo_cover/2_yanwo_conver_data/label/202.png"
if img is None:
    raise ValueError("Image not found! Please check the file path.")

# 根据比例调整图像大小
small_img = cv2.resize(img, None, fx=scale, fy=scale)
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", draw_circle, img.shape)

while True:
    # 显示缩放后的图像，并动态绘制所有圆
    display_img = small_img.copy()
    for (cx, cy, r) in circles:
        # 根据缩放比例在小图上画出所有已记录的圆
        cv2.circle(display_img, (int(cx * scale), int(cy * scale)), int(r * scale), (0, 255, 0), 2)
    if drawing:
        # 绘制当前正在画的圆
        cv2.circle(display_img, center, radius, (0, 255, 0), 2)
    cv2.imshow("Image", display_img)

    # 按 'q' 键退出
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # 按 's' 键保存掩码图像
        mask = create_final_mask(img.shape)
        #cv2.imshow("Mask", cv2.resize(mask, None, fx=scale, fy=scale))  # 显示最终掩码
        cv2.imwrite(la, mask)  # 保存最终掩码图像
        print("Mask saved success")

cv2.destroyAllWindows()
