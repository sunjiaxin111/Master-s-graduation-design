import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']

# 相关数据
x = [2, 4, 6, 8]
criteo_logloss = [0.46632615, 0.46586333, 0.47230573, 0.47971707]
avazu_logloss = [0.39806155, 0.40538578, 0.40406571, 0.40678924]
criteo_auc = [0.75521102, 0.75614338, 0.74894022, 0.75247951]
avazu_auc = [0.76528282, 0.76601039, 0.76632607, 0.76505791]

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
plt.xlabel("隐向量的向量长度embedding_size")  # X轴标签
plt.ylabel("logloss")  # Y轴标签
plt.title("不同embedding_size下AFFM模型的logloss")  # 图标题
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
plt.xlabel("隐向量的向量长度embedding_size")  # X轴标签
plt.ylabel("auc")  # Y轴标签
plt.title("不同embedding_size下AFFM模型的auc")  # 图标题
plt.xticks(x, x)

plt.legend()
plt.show()  # 显示图
