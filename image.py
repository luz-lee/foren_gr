import cv2
import numpy as np
from utils import resize_image, text_to_image

def apply_image_watermark(image, text):
    height, width, _ = image.shape
    img_wm = np.zeros((height, width), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    thickness = 3
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    cv2.putText(img_wm, text, (text_x, text_y), font, font_scale, 255, thickness)

    img_f = np.fft.fft2(image, axes=(0, 1))
    wm_f = np.fft.fft2(img_wm, axes=(0, 1))

    result_f = img_f + (wm_f[:, :, np.newaxis] * 0.1)
    result = np.fft.ifft2(result_f, axes=(0, 1))
    result = np.abs(result).clip(0, 255).astype(np.uint8)

    return result
