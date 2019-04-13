import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']

# 相关数据
x = [0.1, 0.3, 0.5, 0.7, 0.9]
criteo_logloss = [0.48137272, 0.46586333, 0.46680351, 0.46639042, 0.46707553]
avazu_logloss = [0.42036851, 0.40872374, 0.40305382, 0.39985096, 0.39959078]
criteo_auc = [0.74843653, 0.75614338, 0.75588021, 0.75554126, 0.75528870]
avazu_auc = [0.75790215, 0.76111267, 0.76143274, 0.76144173, 0.76146043]

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
plt.xlabel("二阶项的dropout率dropout_2")  # X轴标签
plt.ylabel("logloss")  # Y轴标签
plt.title("不同dropout_2下AFFM模型的logloss")  # 图标题
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
plt.xlabel("二阶项的dropout率dropout_2")  # X轴标签
plt.ylabel("auc")  # Y轴标签
plt.title("不同dropout_2下AFFM模型的auc")  # 图标题
plt.xticks(x, x)

plt.legend()
plt.show()  # 显示图
