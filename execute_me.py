import argparse
import json
import os

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
        """计算曲线的局部斜率，并只保留有效区域"""
        slopes = []
        x_coords = []

        # 使用滑动窗口计算局部斜率
        for i in range(0, len(points) - window_size, window_size // 2):
            window = points[i : i + window_size]
            x, y = window[:, 0], window[:, 1]

            # 使用线性回归计算斜率
            A = np.vstack([x, np.ones(len(x))]).T
            slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]

            # 只有当斜率的绝对值大于阈值时才保留
            if abs(slope) > 0.01:  # 阈值，可调
                slopes.append(slope)
                x_coords.append(np.mean(x))

        # 确保至少有一些有效点
        if len(slopes) < 2:
            raise ValueError("未检测到足够的有效斜率区域")

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
            adjustment_factor = 1  # 调整系数，一般不用动
            cumulative_shift -= (
                slope_diff * adjustment_factor * (x_slopes[1] - x_slopes[0])
            )

            key_points_original.append(x)
            key_points_target.append(x + cumulative_shift)

        # 创建映射表
        grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        map_x = grid_x.astype(np.float32)

        # 只需计算一次插值映射，然后广播到所有行
        map_x[:] = np.interp(
            np.arange(width),  # 原始x坐标点
            key_points_original,  # 已知的原始关键点x坐标
            key_points_target,  # 对应的目标关键点x坐标
        )

        self.correction_map = (map_x, grid_y.astype(np.float32))
        return x_slopes, slopes  # 返回用于可视化的数据

    def correct_image(self, image):
        """使用映射表校正图像并裁剪无效区域"""
        # 使用映射表进行校正
        corrected = cv2.remap(
            image,
            self.correction_map[0],
            self.correction_map[1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        # 获取有效区域的范围
        valid_start = int(self.correction_map[0].min())
        valid_end = int(self.correction_map[0].max())

        # 裁剪图像
        cropped = corrected[:, valid_start:valid_end]

        # 调整大小到原始宽度
        if cropped.shape[1] != image.shape[1]:
            cropped = cv2.resize(cropped, (image.shape[1], image.shape[0]))

        return cropped

    def process_image(self, image):
        """完整的图像处理流程"""
        self.curve_points = self.detect_curve(image)
        x_slopes, slopes = self.create_correction_map(image.shape)
        corrected = self.correct_image(image)
        return corrected, self.curve_points, (x_slopes, slopes)

    def save_correction_map(self, save_path):
        """保存校正映射表"""
        if self.correction_map is None:
            raise ValueError("校正映射表尚未生成")

        # 将映射表转换为可序列化的格式
        map_data = {
            "map_x": self.correction_map[0].tolist(),
            "map_y": self.correction_map[1].tolist(),
        }

        with open(save_path, "w") as f:
            json.dump(map_data, f)

    def load_correction_map(self, load_path):
        """加载校正映射表"""
        with open(load_path, "r") as f:
            map_data = json.load(f)

        self.correction_map = (
            np.array(map_data["map_x"], dtype=np.float32),
            np.array(map_data["map_y"], dtype=np.float32),
        )


def generate_reference_data(reference_path, output_dir):
    """生成参考数据并保存分析图"""
    # 读取参考图像
    reference_image = cv2.imread(reference_path)
    if reference_image is None:
        raise FileNotFoundError(f"无法读取参考图像: {reference_path}")

    # 创建校正器并处理图像
    corrector = ImageCorrector()
    corrected, points, slope_info = corrector.process_image(reference_image)
    x_slopes, slopes = slope_info

    # 保存校正映射表
    map_path = os.path.join(output_dir, "correction_map.json")
    corrector.save_correction_map(map_path)

    # 生成分析图并保存
    plt.figure(figsize=(12, 12))

    # 1. 原始图像和检测到的曲线
    plt.subplot(221)
    plt.imshow(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB))
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
        np.arange(reference_image.shape[1]),
        map_x[reference_image.shape[0] // 2],
        "b-",
        label="映射关系",
    )
    plt.plot(
        [0, reference_image.shape[1]],
        [0, reference_image.shape[1]],
        "r--",
        label="原始位置",
    )
    plt.title("水平方向映射关系")
    plt.xlabel("原始x坐标")
    plt.ylabel("映射后x坐标")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reference_analysis.png"))
    plt.close()

    return corrector


def process_images(corrector, input_dir, output_dir):
    """处理输入目录中的所有图像"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            # 读取图像
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"无法读取图像: {image_path}")
                continue

            # 校正图像
            corrected = corrector.correct_image(image)

            # 保存校正后的图像
            output_path = os.path.join(output_dir, f"corrected_{filename}")
            cv2.imwrite(output_path, corrected)
            print(f"已处理: {filename}")


if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="图像校正工具")
    parser.add_argument(
        "--reference_dir",
        default="./reference_image",
        help="参考图像目录路径, 参考图像需命名为reference.jpg",
    )
    parser.add_argument("--raw_dir", default="./raw_images", help="待处理图像目录路径")
    parser.add_argument(
        "--output_dir", default="./corrected_images", help="输出图像目录路径"
    )
    args = parser.parse_args()

    # 设置中文显示
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 确保目录存在
    os.makedirs(args.reference_dir, exist_ok=True)
    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # 检查或生成校正映射表
    map_path = os.path.join(args.reference_dir, "correction_map.json")
    reference_path = os.path.join(args.reference_dir, "reference.jpg")

    corrector = ImageCorrector()
    if os.path.exists(map_path):
        print("加载已有校正映射表...")
        corrector.load_correction_map(map_path)
    else:
        print("生成新的校正映射表...")
        if not os.path.exists(reference_path):
            raise FileNotFoundError("参考图像不存在！")
        corrector = generate_reference_data(reference_path, args.reference_dir)

    # 处理图像
    process_images(corrector, args.raw_dir, args.output_dir)
