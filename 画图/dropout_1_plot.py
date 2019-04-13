import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']

# 相关数据
x = [0.1, 0.3, 0.5, 0.7, 0.9]
criteo_logloss = [0.47990304, 0.47914524, 0.46586333, 0.46733858, 0.46793087]
avazu_logloss = [0.45851378, 0.40043207, 0.39873953, 0.39893321, 0.39976130]
criteo_auc = [0.75019412, 0.75407946, 0.75614338, 0.75368287, 0.75334768]
avazu_auc = [0.76488331, 0.76530343, 0.76531039, 0.76324534, 0.76048314]

plt.figure(figsize=(8, 3))

# 绘制logloss相关图片
plt.subplot(1, 2, 1)
plt.plot(x, criteo_logloss, label='criteo数据集', color='b', marker='o',
         markerfacecolor='b', markersize=5)
plt.plot(x, avazu_logloss, label='avazu数据集', color='darkorange', marker='o',
         markerfacecolor='darkorange', markersize=5)
# 设置数字标签
for a, b in zip(x, criteo_logloss):
    plt.text(a, b + 0.0001, '%.4f' % b, ha='center', va='bottom', fontsize=11)
for a, b in zip(x, avazu_logloss):
    plt.text(a, b + 0.0001, '%.4f' % b, ha='center', va='bottom', fontsize=11)
plt.xlabel("一阶项的dropout率dropout_1")  # X轴标签
plt.ylabel("logloss")  # Y轴标签
plt.title("不同dropout_1下AFFM模型的logloss")  # 图标题
plt.xticks(x, x)
# plt.ylim((0.3975, 0.49))
plt.legend()

# 绘制auc相关图片
plt.subplot(1, 2, 2)
plt.plot(x, criteo_auc, label='criteo数据集', color='b', marker='o',
         markerfacecolor='b', markersize=5)
plt.plot(x, avazu_auc, label='avazu数据集', color='darkorange', marker='o',
         markerfacecolor='darkorange', markersize=5)
# 设置数字标签
for a, b in zip(x, criteo_auc):
    plt.text(a, b + 0.00001, '%.4f' % b, ha='center', va='bottom', fontsize=11)
for a, b in zip(x, avazu_auc):
    plt.text(a, b + 0.00001, '%.4f' % b, ha='center', va='bottom', fontsize=11)
plt.xlabel("一阶项的dropout率dropout_1")  # X轴标签
plt.ylabel("auc")  # Y轴标签
plt.title("不同dropout_1下AFFM模型的auc")  # 图标题
plt.xticks(x, x)

plt.legend()
plt.show()  # 显示图
