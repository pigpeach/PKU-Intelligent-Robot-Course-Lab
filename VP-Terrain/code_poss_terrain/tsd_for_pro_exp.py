import numpy as np

# 生成一个包含 0 到 1 的均匀间隔数字序列
start_points = np.arange(0, 1.05, 0.05)

# 生成一个包含每个区间起止点的 np 数组
intervals = np.array([start_points[:-1], start_points[1:]]).T

# 打印生成的 np 数组
print(intervals)


# pro_pre=np.loadtxt('./cpv/dark_pre_8857')

y_pre=np.loadtxt('./cpv/test_pro_exp_cos_dis_0.3_norm_pre',delimiter=",")
conf_pre=np.loadtxt('./cpv/test_pro_exp_cos_dis_0.3_norm_max',delimiter=",")
y_res=np.loadtxt('./cpv/test_y_res_8659.txt',delimiter=",")

y_pre_2=np.loadtxt('./cpv/dark_pro_exp_cos_dis_0.3_norm_pre',delimiter=",")
conf_pre_2=np.loadtxt('./cpv/dark_pro_exp_cos_dis_0.3_norm_max',delimiter=",")
y_res_2=np.loadtxt('./cpv/dark_y_res_8857.txt',delimiter=",")

def calculate_tsd(conf_pre, y_res, y_pre, save_):
    conf_pre = conf_pre
    y_res = y_res
    y_pre = y_pre

    unique_values, counts = np.unique(y_res, return_counts=True)
    for i in range(len(unique_values)):
        print(f"{unique_values[i]}: {counts[i]}")
    print(counts.shape)

    # for save_ in range (7):
    ysd_matrix = np.zeros([7,20])
    print(y_pre.shape)
    for index in range(20): # 20列
        for i in range(3912):   # i是第几条数据
            for j in range(7):  # j是预测类别
                # for k in range(7):
                    if y_res[i] == j and y_pre[i] == save_ and conf_pre[i]<=intervals[index][1] and conf_pre[i]>intervals[index][0]:
                        ysd_matrix[j][index] += 1

    print(ysd_matrix)
    return ysd_matrix, counts

for save_index in range (7):    # 指定一个类别
    matrix_test, counts_test = calculate_tsd(conf_pre, y_res, y_pre, save_index)
    matrix_dark, counts_dark = calculate_tsd(conf_pre_2, y_res_2, y_pre_2, save_index)

    mix_matrix = matrix_test + matrix_dark
    counts = counts_test + counts_dark

    # mix_matrix = matrix_dark
    # counts = counts_dark

    # mix_matrix = matrix_test
    # counts = counts_test

    tsd = np.zeros([7,20])
    for i in range(mix_matrix.shape[0]):
        tsd[i] = mix_matrix[i] / counts[i]
    np.savetxt('./cpv/mix_exp_8967_tsd/tsd_{}.txt'.format(save_index),tsd,fmt='%.4f',delimiter=",")