import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import os

# 获取所有可用字体
all_fonts = [f.name for f in fm.fontManager.ttflist]

# 检查目标字体
target_fonts = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'Heiti SC']
available = [font for font in target_fonts if font in all_fonts]

print("可用的中文字体:", available)


# 设置中文显示
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 备用方案
# plt.rcParams['axes.unicode_minus'] = False
#
# # 直接指定字体路径（选择一种）
# font_paths = [
#     '/System/Library/Fonts/PingFang.ttc',      # 苹方字体
#     '/System/Library/Fonts/Hiragino Sans GB.ttc',  # 冬青黑体
#     '/System/Library/Fonts/STHeiti Medium.ttc'  # 华文黑体
# ]
#
# # 选择第一个可用的字体
# for path in font_paths:
#     if os.path.exists(path):
#         font_prop = fm.FontProperties(fname=path)
#         plt.rcParams['font.family'] = 'sans-serif'
#         plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
#         print(f"使用字体: {font_prop.get_name()}")
#         break
# else:
#     print("警告: 未找到中文字体，使用备用方案")
#     plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
#
# 打印可用字体列表
print("可用中文字体:")
for font in sorted(set([f.name for f in mpl.font_manager.fontManager.ttflist])):
    if any(char in font for char in ['宋', '黑', '楷', '仿宋', 'Hei', 'Song', 'Kai', 'Fang']):
        print(f"  - {font}")

# 测试中文显示
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['PingFang SC']
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']  # 或 ['PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

plt.figure()
plt.title("中文标题测试 - 苹果电脑")
plt.xlabel("X轴标签")
plt.ylabel("Y轴标签")
plt.plot([1, 2, 3], [1, 4, 9])
plt.text(1.5, 5, "中文文本示例", fontsize=12)
plt.grid(True)
plt.show()