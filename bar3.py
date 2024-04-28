import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np


def computeOur(N,F,E): # whole graph?
    rot=0
    mulc=0
    mulp=0
    addc=0
    addp=0

    M=4096
    f=M
    F2=32
    F3=16

    # E=min(E,2*N)
    
    # F=300

    total_n=N
    total_f=int(np.ceil(F/f))
    total_f2=int(np.ceil(F2/f))
    total_f3=int(np.ceil(F3/f))

    n_cipher=int(np.ceil(F*F2/M))

    # layer 1
    # Compute XW
    rot+=total_n*n_cipher*int(np.ceil(np.log2(M/F2)))  #*F2
    mulp+=total_n*n_cipher
    addc+=total_n*n_cipher*(int(np.ceil(np.log2(M/F2)))+1)

    # Compute AX
    mulc+=E # total_n+E 提前乘好a_ii
    addc+=E

    # activation
    mulc+=total_n*total_f2

    #layer2
    # Compute AX
    mulc+=E*total_f2
    addc+=E*total_f2

    # Compute XW
    rot+=total_n*F2
    mulp+=total_n*F2
    addc+=total_n*F2


    # rot=86656
    # addc=116444
    # mulc=2708
    # mulp=119152

    # print(rot,mulc,mulp,addc,addp)
    # print('Ratio '+str(rot/(N*F2/np.sqrt(M))))


    time_l=[96, 124, 290, 686+1491,4894]
    time_l=[96,92,275+1185,612+5504+1185,6041]

    rot_time=time_l[4]*rot/1000/1000
    mulc_time=time_l[3]*mulc/1000/1000
    mulp_time=time_l[2]*mulp/1000/1000
    addc_time=time_l[1]*addc/1000/1000
    addp_time=time_l[0]*addp/1000/1000
    total_time=rot_time+mulc_time+mulp_time+addc_time+addp_time

    # return total_time
    return mulc_time

dataset_name_list = ['Pumbed','CoauthorCS','Flickr','CoraFull','AmazonComputer','CoauthorPhysics','RomanEmpire','AmazonRatings','Minesweeper','Questions','DBLP','DeezerEurope']
dataset_name = dataset_name_list[6]
file_path = './performance/he_seg_'+dataset_name+'.txt'
# file_path = './performance/AmazonCoBuyPhoto_mean.txt'

with open(file_path, 'r') as file:
    lines = file.readlines()

origin_auc=float(lines[4].split(',')[0])
origin_q=float(lines[2].split(',')[1])
epsilon_list0 = list(map(float, lines[-5].split(',')))
q_list0 = list(map(float, lines[-4].split(',')))
auc_list0 = list(map(float, lines[-3].split(',')))

epsilon_list=[]
q_list=[]
auc_list=[]
q_diff=(q_list0[1]-q_list0[0])*1.7 #调整阈值,越低保留的点越多
last_q=q_list0[0]
for i in range(1,len(q_list0)):
    if q_list0[i]-last_q>q_diff:
        epsilon_list.append(epsilon_list0[i])
        q_list.append(q_list0[i])
        auc_list.append(auc_list0[i])
        last_q=q_list0[i]

q_list=[origin_q]+q_list
auc_list=auc_list

time_list=[computeOur(float(lines[2].split(',')[0]),float(lines[2].split(',')[2]),i) for i in q_list]

time_list=[i/time_list[0] for i in time_list[1:]]
q_list=q_list[1:]

print(len(epsilon_list), len(q_list), len(auc_list))
print(np.min(time_list), np.max(time_list))

# print(epsilon_list[16],auc_list[16],time_list[16])

#find min diff of q_list
min_diff = 100000000
min_diff_index = -1
for i in range(1, len(time_list)):
    diff = q_list[i] - q_list[i-1]
    if diff < min_diff:
        min_diff = diff
        min_diff_index = i


# 假设数据
size = len(epsilon_list)  # 示例数据长度
x = [i/1000 for i in q_list]
print(x)
y3 = time_list  # 用于柱状图的数据
y_line1 = auc_list  # 第一条折线图的数据

fig, ax1 = plt.subplots(figsize=(5.5, 3.2))

# 柱状图
width = min_diff/1000*0.8
bars3 = ax1.bar(x, y3, width=width, label='Lat./Original Lat.', color='white', zorder=10, edgecolor='black', linewidth=1, hatch='//////')

# 设置左侧y轴标签
ax1.set_ylabel('Latency/Original Lat.', fontsize=12)
ax1.set_ylim((0.5,1))
ax1.yaxis.set_major_locator(MultipleLocator(0.1))
ax1.set_xlabel('# of Edges (k)', fontsize=12)

# 创建第二个y轴用于折线图
ax2 = ax1.twinx()


# 添加两条折线图
# line2, = ax2.plot(x, [origin_auc]*len(x), label='Original AUC', color='black', linestyle='dotted', marker='s',markersize=2)
line1, = ax2.plot(x, y_line1, label='AUC',c='r',linestyle='--',marker='o',markersize=3)

line2= ax2.axhline(y=origin_auc, color='black', linestyle='dotted', label='Original AUC')

# 设置右侧y轴标签
ax2.set_ylabel('AUC', fontsize=12)
ax2.set_ylim((0.5, 1.))
ax2.yaxis.set_major_locator(MultipleLocator(0.1))

# 合并图例
lines = [bars3, line1,line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center', bbox_to_anchor=(0.5, 1.2), ncol=3, fontsize=11)
plt.title(dataset_name+'Dataset', fontsize=12,loc='center', y=1.35)

# plt.xlim((-width, size - 0.3))  # 调整x轴的范围以适应所有的柱状图
plt.tight_layout()
plt.savefig('./pic/'+dataset_name+'_he_seg2_modified.pdf', bbox_inches='tight')
plt.show()