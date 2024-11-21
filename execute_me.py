import cv2
import matplotlib.pyplot as plt
import numpy as np


class ImageCorrector:
    def __init__(self):
        self.curve_points = None
        self.correction_map = None

    def detect_curve(self, image):
        """检测图像中的曲线，返回曲线上的关键点坐标"""
        # 图像预处理
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        )
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)

        # 连通域分析，找到最大连通域
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        largest_label = max(
            range(1, num_labels), key=lambda i: stats[i, cv2.CC_STAT_AREA]
        )

        # 创建掩码，只保留最大连通域
        curve_mask = np.zeros_like(binary)
        curve_mask[labels == largest_label] = 255

        # 对于每个x坐标，只保留最上面的点
        points = []
        for x in range(curve_mask.shape[1]):
            y_coords = np.where(curve_mask[:, x] == 255)[0]
            if len(y_coords) > 0:
                points.append([x, y_coords[0]])

        points = np.array(points)
        return points[points[:, 0].argsort()]  # 确保点按x坐标排序

    def calculate_local_slopes(self, points, window_size=50):
        """计算曲线的局部斜率"""
        slopes = []
        x_coords = []

        # 使用滑动窗口计算局部斜率
        for i in range(0, len(points) - window_size, window_size // 2):
            window = points[i : i + window_size]
            x, y = window[:, 0], window[:, 1]

            # 使用线性回归计算斜率
            A = np.vstack([x, np.ones(len(x))]).T
            slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]

            slopes.append(slope)
            x_coords.append(np.mean(x))

        return np.array(x_coords), np.array(slopes)

    def create_correction_map(self, image_shape):
        """创建基于局部斜率的校正映射表"""
        height, width = image_shape[:2]
        x_slopes, slopes = self.calculate_local_slopes(self.curve_points)
        mean_slope = np.mean(slopes)

        # 计算关键映射点
        key_points_original = []
        key_points_target = []

        # 使用斜率差的累加来计算位移
        cumulative_shift = 0
        for i in range(len(x_slopes)):
            x = x_slopes[i]
            slope = slopes[i]

            # 计算与平均斜率的差值并累加位移
            slope_diff = slope - mean_slope
            adjustment_factor = 1
            cumulative_shift -= (
                slope_diff * adjustment_factor * (x_slopes[1] - x_slopes[0])
            )

            key_points_original.append(x)
            key_points_target.append(x + cumulative_shift)

        # 创建映射表
        grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        map_x = grid_x.astype(np.float32)

        # 对每一行进行插值映射
        for y in range(height):
            map_x[y] = np.interp(
                np.arange(width), key_points_original, key_points_target
            )

        self.correction_map = (map_x, grid_y.astype(np.float32))
        return x_slopes, slopes  # 返回用于可视化的数据

    def correct_image(self, image):
        """使用映射表校正图像"""
        return cv2.remap(
            image,
            self.correction_map[0],
            self.correction_map[1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    def process_image(self, image):
        """完整的图像处理流程"""
        self.curve_points = self.detect_curve(image)
        x_slopes, slopes = self.create_correction_map(image.shape)
        corrected = self.correct_image(image)
        return corrected, self.curve_points, (x_slopes, slopes)


def main():
    # 设置中文显示
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 读取并处理图像
    image = cv2.imread("img.jpg")
    corrector = ImageCorrector()
    corrected, points, slope_info = corrector.process_image(image)
    x_slopes, slopes = slope_info

    # 创建2x2的子图布局
    plt.figure(figsize=(12, 12))

    # 1. 原始图像和检测到的曲线
    plt.subplot(221)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.scatter(points[:, 0], points[:, 1], c="r", s=1)
    plt.title("原始图像和检测曲线")

    # 2. 局部斜率分布
    plt.subplot(222)
    plt.plot(x_slopes, slopes, "g-", label="局部斜率")
    plt.axhline(y=np.mean(slopes), color="r", linestyle="--", label="平均斜率")
    plt.title("局部斜率分布")
    plt.xlabel("x坐标")
    plt.ylabel("斜率")
    plt.legend()

    # 3. 校正后图像
    plt.subplot(223)
    plt.imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
    plt.title("校正后图像")

    # 4. 映射关系可视化
    plt.subplot(224)
    map_x = corrector.correction_map[0]
    plt.plot(
        np.arange(image.shape[1]), map_x[image.shape[0] // 2], "b-", label="映射关系"
    )
    plt.plot([0, image.shape[1]], [0, image.shape[1]], "r--", label="原始位置")
    plt.title("水平方向映射关系")
    plt.xlabel("原始x坐标")
    plt.ylabel("映射后x坐标")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
