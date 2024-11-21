import cv2
import matplotlib.pyplot as plt
import numpy as np


class ImageCorrector:
    def __init__(self):
        self.curve_points = None
        self.correction_map = None

    def detect_curve(self, image):
        """
        检测图像中的曲线
        返回曲线上的关键点坐标
        """
        # 图像预处理
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        )
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)

        # 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        # 找到最大的连通域
        largest_label = 1
        largest_area = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > largest_area:
                largest_area = area
                largest_label = i

        # 创建掩码，只保留最大连通域
        curve_mask = np.zeros_like(binary)
        curve_mask[labels == largest_label] = 255

        # 对于每个x坐标，只保留最上面的点
        points = []
        for x in range(curve_mask.shape[1]):
            # 获取当前列中值为255的点的y坐标
            y_coords = np.where(curve_mask[:, x] == 255)[0]
            if len(y_coords) > 0:
                # 只取最上面的点（y坐标最小的点）
                points.append([x, y_coords[0]])

        points = np.array(points)

        # 确保点按x坐标排序
        points = points[points[:, 0].argsort()]

        return points

    def calculate_local_slopes(self, points, window_size=50):
        """
        计算局部斜率
        points: 检测到的曲线点
        window_size: 计算局部斜率时使用的窗口大小
        """
        slopes = []
        x_coords = []

        # 确保点按x坐标排序
        sorted_points = points[points[:, 0].argsort()]

        # 使用滑动窗口计算局部斜率
        for i in range(0, len(sorted_points) - window_size, window_size // 2):
            window = sorted_points[i : i + window_size]
            x = window[:, 0]
            y = window[:, 1]

            # 使用线性回归计算斜率
            A = np.vstack([x, np.ones(len(x))]).T
            slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]

            slopes.append(slope)
            x_coords.append(np.mean(x))

        return np.array(x_coords), np.array(slopes)

    def create_correction_map(self, image_shape):
        """
        创建基于局部斜率的校正映射表，使用插值实现平滑缩放
        """
        height, width = image_shape[:2]

        # 计算局部斜率
        x_slopes, slopes = self.calculate_local_slopes(self.curve_points)
        mean_slope = np.mean(slopes)

        # 计算关键映射点
        key_points_original = []  # 原始x坐标
        key_points_target = []  # 目标x坐标

        # 使用斜率差的累加来计算位移
        cumulative_shift = 0
        for i in range(len(x_slopes)):
            x = x_slopes[i]
            slope = slopes[i]

            # 计算与平均斜率的差值
            slope_diff = slope - mean_slope

            # 累加位移（斜率差越大，位移越大）
            # 可以调整这个系数来控制校正强度
            adjustment_factor = 1  # 调整系数
            cumulative_shift -= (
                slope_diff * adjustment_factor * (x_slopes[1] - x_slopes[0])
            )

            key_points_original.append(x)
            key_points_target.append(x + cumulative_shift)

        # 创建映射表
        grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        map_x = grid_x.astype(np.float32)
        map_y = grid_y.astype(np.float32)

        # 对每一列进行插值映射
        for y in range(height):
            map_x[y] = np.interp(
                np.arange(width), key_points_original, key_points_target
            )

        self.correction_map = (map_x, map_y)
        return self.correction_map

    def correct_image(self, image):
        """
        使用映射表校正图像，并进行宽度调整
        """
        if self.correction_map is None:
            self.create_correction_map(image.shape)

        map_x, map_y = self.correction_map

        # 应用校正
        corrected = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        # 获取校正后图像的宽度
        corrected_width = corrected.shape[1]
        original_width = image.shape[1]

        if corrected_width != original_width:
            # 修正：使用正确的源和目标坐标范围
            src_x = np.linspace(0, original_width - 1, corrected_width)  # 修改这里
            dst_x = np.arange(original_width)  # 修改这里

            # 对每一行进行重采样
            resized = np.zeros_like(image)
            for y in range(image.shape[0]):
                # 对每个通道进行插值
                for c in range(image.shape[2] if len(image.shape) > 2 else 1):
                    if len(image.shape) > 2:
                        resized[y, :, c] = np.interp(dst_x, src_x, corrected[y, :, c])
                    else:
                        resized[y, :] = np.interp(dst_x, src_x, corrected[y, :])

            return resized

        return corrected

    def get_corrected_curve_points(self):
        """
        获取校正后的曲线点坐标
        """
        if self.correction_map is None:
            return None

        map_x, _ = self.correction_map

        # 获取原始曲线点的新位置
        corrected_points = []
        for x, y in self.curve_points:
            new_x = map_x[int(y), int(x)]
            corrected_points.append([new_x, y])

        return np.array(corrected_points)

    def get_target_curve_points(self, image_shape):
        """
        获取目标曲线点（使用平均斜率的直线）
        """
        x_slopes, slopes = self.calculate_local_slopes(self.curve_points)
        mean_slope = np.mean(slopes)

        # 使用第一个点作为起点
        start_point = self.curve_points[0]
        x_coords = np.arange(image_shape[1])
        y_coords = start_point[1] + mean_slope * (x_coords - start_point[0])

        return np.column_stack((x_coords, y_coords))

    def process_image(self, image):
        """
        完整的图像处理流程
        """
        # 1. 检测曲线
        self.curve_points = self.detect_curve(image)

        # 2. 计算局部斜率（用于可视化）
        x_slopes, slopes = self.calculate_local_slopes(self.curve_points)

        # 3. 创建校正映射
        self.create_correction_map(image.shape)

        # 4. 校正图像
        corrected = self.correct_image(image)

        # 5. 获取各种曲线点（用于可视化）
        x_coords = np.arange(image.shape[1])
        y_coords = np.interp(x_coords, self.curve_points[:, 0], self.curve_points[:, 1])
        curve_points = np.column_stack((x_coords, y_coords))

        corrected_curve = self.get_corrected_curve_points()
        target_curve = self.get_target_curve_points(image.shape)

        return (
            corrected,
            self.curve_points,
            curve_points,
            (x_slopes, slopes),
            corrected_curve,
            target_curve,
        )


def main():
    # 设置中文显示
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 读取图像
    image = cv2.imread("img.jpg")

    # 创建校正器实例
    corrector = ImageCorrector()

    # 处理图像并获取所有中间结果
    corrected, points, curve_points, slope_info, corrected_curve, target_curve = (
        corrector.process_image(image)
    )
    x_slopes, slopes = slope_info

    # 创建2x3的子图布局
    plt.figure(figsize=(18, 12))

    # 1. 原始图像和检测到的曲线
    plt.subplot(231)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.scatter(points[:, 0], points[:, 1], c="r", s=1, label="检测点")
    plt.plot(curve_points[:, 0], curve_points[:, 1], "b-", label="插值曲线")
    plt.title("原始图像和检测曲线")
    plt.legend()

    # 2. 局部斜率分布
    plt.subplot(232)
    plt.plot(x_slopes, slopes, "g-", label="局部斜率")
    plt.axhline(y=np.mean(slopes), color="r", linestyle="--", label="平均斜率")
    plt.title("局部斜率分布")
    plt.xlabel("x坐标")
    plt.ylabel("斜率")
    plt.legend()

    # 3. 映射关系可视化
    plt.subplot(233)
    map_x = corrector.correction_map[0]
    plt.plot(
        np.arange(image.shape[1]), map_x[image.shape[0] // 2], "b-", label="映射关系"
    )
    plt.plot([0, image.shape[1]], [0, image.shape[1]], "r--", label="原始位置")
    plt.title("水平方向映射关系")
    plt.xlabel("原始x坐标")
    plt.ylabel("映射后x坐标")
    plt.legend()

    # 4. 校正后图像
    plt.subplot(234)
    plt.imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
    plt.title("校正后图像")

    # 5. 曲线校正效果对比
    plt.subplot(235)
    plt.scatter(points[:, 0], points[:, 1], c="r", s=1, label="原始曲线")
    plt.scatter(
        corrected_curve[:, 0], corrected_curve[:, 1], c="g", s=1, label="校正后曲线"
    )
    plt.plot(target_curve[:, 0], target_curve[:, 1], "b--", label="目标曲线")
    plt.title("曲线校正效果对比")
    plt.legend()

    # 6. 校正前后对比
    plt.subplot(236)
    plt.imshow(cv2.cvtColor(np.hstack((image, corrected)), cv2.COLOR_BGR2RGB))
    plt.title("校正前后对比")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
