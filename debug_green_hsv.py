"""调试：分析图片中绿色涂抹的 HSV 值分布。"""
import cv2
import numpy as np

path = r"D:\codes\work-projects\CAM\imgs\C_AHX20180625M66_002_bbox.jpg"
bgr = cv2.imread(path)
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

# 先大致框定左上方区域（你描述绿色在左上方 1/4）
h, w = hsv.shape[:2]
# 取左上 1/2 区域作为候选区，避免整张图统计被背景淹没
crop = hsv[0:h//2, 0:w//2]

# 对候选区，找 H 通道中等偏绿的部分（常见绘图绿 H 在 35~90）
h_channel = crop[:, :, 0]
# 只取饱和度较高的像素（排除灰白背景）
mask = crop[:, :, 1] > 40
h_values = h_channel[mask]

print(f"左上区域高饱和像素数: {len(h_values)}")
print(f"H 值范围: {h_values.min()} ~ {h_values.max()}")
print(f"H 值均值/中位数: {h_values.mean():.1f} / {np.median(h_values):.1f}")

# 进一步，在左上区里，H 在 30~100 之间的像素的 S/V 统计
greenish = crop[(crop[:, :, 0] >= 30) & (crop[:, :, 0] <= 100) & (crop[:, :, 1] > 30)]
if len(greenish) > 0:
    print(f"\n左上区 '绿-ish' 像素数: {len(greenish)}")
    print(f"H: {greenish[:, 0].min():.0f}~{greenish[:, 0].max():.0f}, 均值={greenish[:, 0].mean():.1f}")
    print(f"S: {greenish[:, 1].min():.0f}~{greenish[:, 1].max():.0f}, 均值={greenish[:, 1].mean():.1f}")
    print(f"V: {greenish[:, 2].min():.0f}~{greenish[:, 2].max():.0f}, 均值={greenish[:, 2].mean():.1f}")

# 保存当前默认参数提取的 mask 供目视检查
from extract_green_bbox import extract_green_boxes, save_box_artifacts
boxes = extract_green_boxes(path)
print(f"\n当前默认参数提取到: {len(boxes)} 个框")
if boxes:
    print(f"框坐标: {boxes}")

# 额外：用更宽的 H 范围试一次
mask_wide = cv2.inRange(hsv, np.array([30, 20, 20]), np.array([100, 255, 255]))
ys, xs = np.where(mask_wide > 0)
if len(xs) > 0:
    print(f"\n放宽 H 到 30~100 后，绿色像素包围框: ({xs.min()},{ys.min()}) - ({xs.max()},{ys.max()})")

# 保存 mask 调试图
cv2.imwrite(r"D:\codes\work-projects\CAM\outputs\debug_mask_default.png", cv2.inRange(hsv, np.array([35,40,40]), np.array([85,255,255])))
cv2.imwrite(r"D:\codes\work-projects\CAM\outputs\debug_mask_wide.png", mask_wide)
print(r"\n调试 mask 已保存到 outputs\debug_mask_default.png 和 debug_mask_wide.png")
