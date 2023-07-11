im_path = '/media/zou/D/000003072.jpg'
import matplotlib.pyplot as plt
# load
import core
import torch
from PIL import Image
import numpy as np


def bilinear_interpolation(img, out_dim):
    src_h, src_w, channels = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]
    print("src_h,src_w= ", src_h, src_w)
    print("dst_h,dst_w= ", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 根据几何中心重合找出目标像素的坐标
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5
                # 找出目标像素最邻近的四个点
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # 代入公式计算
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img

img = plt.imread('/media/zou/D/000003072.jpg')
print('img',img.shape)

img1 = bilinear_interpolation(img,[128,128])
print('img1',img1.shape)
#读取后的dtype为float32（属于numpy），范围是[0,1]
#读取后的形状是（H,W,C），RGB
# show
plt.imshow(img)
plt.imshow(img1)
plt.axis('off')
plt.show()
# save
plt.imsave('/media/zou/D/lr000003072.jpg',img1)
