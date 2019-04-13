# # criteo数据集
# import matplotlib.pyplot as plt
# from pylab import mpl
#
# mpl.rcParams['font.sans-serif'] = ['SimHei']
#
# # X轴，Y轴数据
# x = ['无交叉熵和正例率特征', '交叉熵特征', '正例率特征', '交叉熵和正例率特征']
# index = list(range(4))
# criteo_y = [0.75279627, 0.75448256, 0.75459841, 0.75508891]
#
# bar_width = 0.5
# plt.bar(index, criteo_y, bar_width, color='b')
#
# plt.xlabel("不同特征")  # X轴标签
# plt.ylabel("auc")  # Y轴标签
# plt.title("criteo数据集不同特征下FFM模型的auc")  # 图标题
#
# for a, b in zip(index, criteo_y):
#     plt.text(a, b + 0.00001, '%.4f' % b, ha='center', va='bottom', fontsize=11)
#
# plt.xticks(index, x)
# plt.ylim((0.75, 0.756))
# plt.show()  # 显示图

# avazu数据集
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']

# X轴，Y轴数据
x = ['无交叉熵和正例率特征', '交叉熵特征', '正例率特征', '交叉熵和正例率特征']
index = list(range(4))
criteo_y = [0.76166069, 0.76378009, 0.76429977, 0.76484266]

bar_width = 0.5
plt.bar(index, criteo_y, bar_width, color='darkorange')

plt.xlabel("不同特征")  # X轴标签
plt.ylabel("auc")  # Y轴标签
plt.title("avazu数据集不同特征下FFM模型的auc")  # 图标题

for a, b in zip(index, criteo_y):
    plt.text(a, b + 0.00001, '%.4f' % b, ha='center', va='bottom', fontsize=11)

plt.xticks(index, x)
plt.ylim((0.76, 0.7655))
plt.show()  # 显示图
