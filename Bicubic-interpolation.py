
'''@author: songcongcong'''


# 三次卷积的核函数
def S(x):
    if abs(x) <= 1:
        y = 1 - 2 * np.power(x, 2) + abs(np.power(x, 3))
    elif abs(x) > 1 and abs(x) < 2:
        y = 4 - 8 * abs(x) + 5 * np.power(x, 2) - abs(np.power(x, 3))
    else:
        y = 0
    return y


# 三次卷积插值
def bicubic_interpolation(src, dst_shape):
    # 获取原图维度
    src_height, src_width = src.shape[0], src.shape[1]
    # 计算新图维度 注意channel数要想同
    dst_height, dst_width, channels = dst_shape[0], dst_shape[1], dst_shape[2]

    dst = np.zeros(shape=(dst_height, dst_width, channels), dtype=np.uint8)
    for dst_x in range(dst_height):
        for dst_y in range(dst_width):
            # 寻找源图像对应坐标
            src_x = (dst_x + 0.5) * (src_width / dst_width) - 0.5
            src_y = (dst_y + 0.5) * (src_width / dst_width) - 0.5
            i, j = int(src_x), int(src_y)
            u, v = src_x - i, src_y - j

            # 边界条件
            x1 = min(max(0, i - 1), src_height - 4)
            x2 = x1 + 4
            y1 = min(max(0, j - 1), src_width - 4)
            y2 = y1 + 4

            # 计算双三次插值
            A = np.array([S(u + 1), S(u), S(u - 1), S(u - 2)])
            C = np.array([S(v + 1), S(v), S(v - 1), S(v - 2)])
            B = src[x1:x2, y1:y2]
            f0 = [A @ B[..., i] @ C.T for i in range(channels)]
            f1 = np.stack(f0)
            f = np.clip(f1, 0, 255)  # 处理一下越界的数据

            # 插值
            dst[dst_x, dst_y, :] = f.astype(np.uint8)
    return dst

im_path = '/media/zou/D/dataset/im0002.jpg'
image = np.array(Image.open(im_path))
image2 = bicubic_interpolation(image,[128,128,3])
image2 = Image.fromarray(image2.astype('uint8')).convert('RGB')
image2.save('/media/zou/D/dataset/BiCubic_interpolation.jpg')
