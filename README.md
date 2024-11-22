# 图像校正工具

这是一个自动检测和校正图像中曲线变形的工具。该工具可以通过分析参考图像中的曲线特征，生成校正映射表，并用于批量校正其他具有相似变形的图像。

## 功能特点

- 自动检测图像中的曲线
- 计算局部斜率并生成校正映射
- 支持批量处理图像
- 生成详细的分析图表
- 可保存和加载校正映射表

## 安装要求

````python
pip install numpy opencv-python matplotlib
````

## 目录结构

````plaintext
.
├── reference_image/     # 存放参考图像
│   └── reference.jpg    # 参考图像
├── raw_images/         # 存放待处理的图像
├── corrected_images/   # 存放校正后的图像
└── execute_me.py       # 主程序
````

## 使用方法

1. 准备参考图像
   - 将包含标准曲线的参考图像命名为 `reference.jpg`
   - 放置在 `reference_image` 目录下

2. 准备待处理图像
   - 将需要校正的图像放入 `raw_images` 目录

3. 运行程序
````bash
python execute_me.py
````

也可以自定义目录路径：
````bash
python execute_me.py --reference_dir ./my_reference --raw_dir ./my_raw --output_dir ./my_output
````

## 输出结果

- 校正后的图像将保存在 `corrected_images` 目录
- 分析图表 `reference_analysis.png` 将生成在参考图像目录，包含：
  - 原始图像和检测到的曲线
  - 局部斜率分布图
  - 校正后的效果图
  - 水平方向映射关系图
- 校正映射表将保存为 `correction_map.json`

## 注意事项

- 参考图像中应当包含清晰的曲线特征
- 支持的图像格式：PNG、JPG、JPEG
- 确保有足够的磁盘空间存储处理后的图像
- 建议使用 Python 3.6 或更高版本

## 工作流程

1. 程序首先检查是否存在校正映射表
2. 如果没有映射表，会使用参考图像生成新的映射表
3. 使用映射表对 `raw_images` 目录中的所有图像进行校正
4. 校正后的图像保存到 `corrected_images` 目录

## 许可证

MIT许可证

````plaintext
MIT License

Copyright (c) [2024] [Horned Axe]

特此授予任何获得本软件和相关文档文件（“软件”）副本的人，免费使用该软件，包括但不限于使用、复制、修改、合并、出版、分发、再许可和/或出售该软件的副本，并允许被提供该软件的人这样做，条件是上述版权声明和本许可声明应包含在所有副本或重要部分中。

该软件是按“原样”提供的，不附带任何明示或暗示的担保，包括但不限于对适销性、特定用途适用性和非侵权的担保。在任何情况下，作者或版权持有人均不对因使用本软件或其他交易而引起的任何索赔、损害或其他责任负责，无论是在合同诉讼、侵权或其他方面。
````
