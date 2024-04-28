import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def computeOurSingle(N,F,E):
    rot=0
    mulc=0
    mulp=0
    addc=0
    addp=0

    M=4096
    f=1
    F2=32
    F3=16

    total_n=N
    total_f=int(np.ceil(F/f))
    total_f2=int(np.ceil(F2/f))
    total_f3=int(np.ceil(F3/f))

    # Compute XW
    rot+=total_f
    mulp+=total_f
    addc+=total_f

    time_l=[96, 124, 290, 686+1491,4894]
    time_l=[96,92,275+1185,612+5504+1185,6041]

    # time_l=[30,58,195+792,464+792+3348,3350] # our CKKS 2^13
    # time_l=[702,1244,3850+15658,7716+15658+134206,141939] # our CKKS 2^15

    rot_time=time_l[4]*rot/1000/1000
    mulc_time=time_l[3]*mulc/1000/1000
    mulp_time=time_l[2]*mulp/1000/1000
    addc_time=time_l[1]*addc/1000/1000
    addp_time=time_l[0]*addp/1000/1000
    total_time=rot_time+mulc_time+mulp_time+addc_time+addp_time

    return mulc_time,rot_time,total_time

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

    # E=min(E,2*N) neighbor
    
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

    print(rot,mulc,mulp,addc,addp)
    # print('Ratio '+str(rot/(N*F2/np.sqrt(M))))


    time_l=[96, 124, 290, 686+1491,4894]
    time_l=[96,92,275+1185,612+5504+1185,6041]

    rot_time=time_l[4]*rot/1000/1000
    mulc_time=time_l[3]*mulc/1000/1000
    mulp_time=time_l[2]*mulp/1000/1000
    addc_time=time_l[1]*addc/1000/1000
    addp_time=time_l[0]*addp/1000/1000
    total_time=rot_time+mulc_time+mulp_time+addc_time+addp_time

    return mulc_time,rot_time,total_time

def computeOur2(N,F,E): # pack multiple x, perform log F rot in XW (ignoring rot random is slow)
    rot=0
    mulc=0
    mulp=0
    addc=0
    addp=0

    M=4096
    f=M
    F2=32
    F3=16

    total_n=N
    total_f=int(np.ceil(F/f))
    total_f2=int(np.ceil(F2/f))
    total_f3=int(np.ceil(F3/f))

    # layer 1
    # Compute AX
    mulc+=E # total_n+E 提前乘好a_ii
    addc+=E

    # Compute XW
    rot+=total_n*int(np.ceil(np.log2(F)))  #*F2
    mulp+=total_n*int(np.ceil(np.log2(F)))
    addc+=total_n*int(np.ceil(np.log2(F)))

    # activation
    mulc+=total_n*total_f

    #layer2
    # Compute AX
    mulc+=E
    addc+=E

    # Compute XW
    rot+=total_n*F2
    mulp+=total_n*F2
    addc+=total_n*F2

    print(rot,mulc,mulp,addc,addp)


    time_l=[96, 124, 290, 686+1491,4894]
    time_l=[96,92,275+1185,612+5504+1185,6041]

    rot_time=time_l[4]*rot/1000/1000
    mulc_time=time_l[3]*mulc/1000/1000
    mulp_time=time_l[2]*mulp/1000/1000
    addc_time=time_l[1]*addc/1000/1000
    addp_time=time_l[0]*addp/1000/1000
    total_time=rot_time+mulc_time+mulp_time+addc_time+addp_time

    return mulc_time,rot_time,total_time


def compute(N,F):
    rot=0
    mulc=0
    mulp=0
    addc=0
    addp=0

    M=4096
    F2=32
    F3=16

    n=int(np.sqrt(2*M))
    n=128
    f=int(M//n)

    total_n=int(np.ceil(N/n))
    total_f=int(np.ceil(F/f))
    total_f2=int(np.ceil(F2/f))
    total_f3=int(np.ceil(F3/f))

    # layer 1

    # Compute XW

    mulp+=total_f*F2*total_n
    addc+=F2*total_n*(total_f-1)

    rot+=2*(f-1)*total_f2*total_n #*total_f
    mulp+=2*(f-1)*total_f2*total_n #*total_f
    addc+=2*(f-1)*total_f2*total_n #*total_f

    # Compute AX

    mulc+=n*total_n*total_n*total_f2
    addc+=(n-1)*total_n*total_n*total_f2
    rot+=(n-1)*total_n*total_f2

    # activation

    mulc+=total_n*total_f2

    #layer2

    # Compute XW

    mulp+=total_f2*F3*total_n
    addc+=F3*total_n*(total_f-1)

    rot+=2*(f-1)*total_f3*total_n #*total_f2
    mulp+=2*(f-1)*total_f3*total_n #*total_f2
    addc+=2*(f-1)*total_f3*total_n #*total_f2

    # Compute AX

    mulc+=n*total_n*total_n*total_f3
    addc+=(n-1)*total_n*total_n*total_f3
    rot+=(n-1)*total_n*total_f3

    print(rot,mulc,mulp,addc,addp)


    time_l=[96, 124, 290, 686+1491,4894]
    time_l=[96,92,275+1185,612+5504+1185,6041]

    rot_time=time_l[4]*rot/1000/1000
    mulc_time=time_l[3]*mulc/1000/1000
    mulp_time=time_l[2]*mulp/1000/1000
    addc_time=time_l[1]*addc/1000/1000
    addp_time=time_l[0]*addp/1000/1000
    total_time=rot_time+mulc_time+mulp_time+addc_time+addp_time

    return mulc_time,rot_time,total_time

def computeIA(N,F):
    rot=0
    mulc=0
    mulp=0
    addc=0
    addp=0

    M=4096
    F2=32
    F3=16

    # F=300

    n=int(np.sqrt(2*M))
    n=128
    f=int(M//n)

    total_n=int(np.ceil(N/n))
    total_f=int(np.ceil(F/f))
    total_f2=int(np.ceil(F2/f))
    total_f3=int(np.ceil(F3/f))

    # layer 1

    # Compute XW

    mulp+=total_f*F2*total_n
    addc+=F2*total_n*(total_f-1)

    rot+=2*(f-1)*total_f2*total_n #*total_f
    mulp+=2*(f-1)*total_f2*total_n #*total_f
    addc+=2*(f-1)*total_f2*total_n #*total_f

    # Compute AX

    mulc+=n*total_n*total_n*total_f2
    addc+=(n-1)*total_n*total_n*total_f2
    rot+=(n-1)*total_n*total_f2

    # activation

    mulc+=total_n*total_f2

    #layer2

    # Compute XW

    mulp+=total_f2*F3*total_n
    addc+=F3*total_n*(total_f-1)

    rot+=2*(f-1)*total_f3*total_n #*total_f2
    mulp+=2*(f-1)*total_f3*total_n #*total_f2
    addc+=2*(f-1)*total_f3*total_n #*total_f2

    # IA
    # total_n=
    rot+=int(np.ceil(total_n/2))*total_f2
    addc+=int(np.ceil(total_n/2))*total_f2

    # Compute AX

    mulc+=n*int(np.ceil(total_n/2))*total_n*total_f3
    addc+=(n-1)*int(np.ceil(total_n/2))*total_n*total_f3
    rot+=(n-1)*int(np.ceil(total_n/2))*total_f3

    print(rot,mulc,mulp,addc,addp)
    # print('Ratio '+str(rot/(N*F2/np.sqrt(M))))


    time_l=[96, 124, 290, 686+1491,4894]
    time_l=[96,92,275+1185,612+5504+1185,6041]
    # time_l=[96,92,180,6440,6200]

    rot_time=time_l[4]*rot/1000/1000
    mulc_time=time_l[3]*mulc/1000/1000
    mulp_time=time_l[2]*mulp/1000/1000
    addc_time=time_l[1]*addc/1000/1000
    addp_time=time_l[0]*addp/1000/1000
    total_time=rot_time+mulc_time+mulp_time+addc_time+addp_time

    return mulc_time,rot_time,total_time

# # Pubmed
# N=19717
# F=500
# E=88651

# # # CiteSeer
# # N=3327
# # F=3703
# # E=9228

# # # Cora
# # N=2708
# # F=1433
# # E=10556

# # # Yelp
# # N=716847
# # F=300

# # Flickr
# N=89250
# F=500
# E=899756

# # CoauthorCS
# N=19793
# F=8710

# # Corafull
# N=18333
# F=6805
# E=163788

# # AmazonCoBuyComputer
# N=13752
# F=767
# E=491722

# # AmazonCoBuyPhotoDataset
# N=7650
# F=745
# E=238163

# # WikiCSDataset
# N=11701
# F=300
# E=431726

# # CoauthorPhysicsDataset
# N=34493
# F=8415
# E=495924

# # Reddit
# N=232965
# F=602
# E=114615892

# # RomanEmpire
# N=22662
# F=300
# E=65854

# # AmazonRatings
# N=24492
# F=300
# E=186100

# # MinesweeperDataset
# N=10000
# F=7
# E=78804

# # Tolokers
# N=11758
# F=10
# E=1038000

# # Questions
# N=48921
# F=301
# E=307080

mpl.rcParams['hatch.linewidth'] = 0.6

# plt.rcParams['text.usetex'] = True

# name = ['Minesweeper','AmazonComputer','CoauthorCS','Pubmed','CoraFull']
# name = ['RomanEmpire','AmazonRatings','CoauthorPhysics','Questions','Flickr']
name = ['CoraFull','CoauthorCS','AmazonComp.','Minesweeper','Pubmed']

size = len(name)

# y=[computeIA(N,F) for N,F in zip([10000,13752,18333,19717,19793],[7,767,6805,500,8710])]
# y=[computeIA(N,F) for N,F in zip([22662,24492,34493,48921,89250],[300,300,8415,301,500])]
y=[computeIA(N,F) for N,F in zip([19793,18333,13752,10000,19717],[8710,6805,767,7,500])]

y1=[i[0]/i[2] for i in y]
y2=[(i[1]+i[0])/i[2] for i in y]
y3=[i[2]/i[2] for i in y]

print('Cora',computeIA(2708,1433)) #1433
print('Pubmed',computeIA(19717,500))
print('CiteSeer',computeIA(3327,3703)) # 3703

print('Our Cora',computeOur(2708,1433,10556)) #1433
print('Our Pubmed',computeOur(19717,500,88651))
print('Our CiteSeer',computeOur(3327,3703,9228)) # 3703

# print('Our Single Cora',computeOurSingle(2708,1433,10556))

print('-----------------------------')
print('CoauthorCS',computeIA(18333,6805)) # 6805
print('Flickr',computeIA(89250,500))
print('CoraFull',computeIA(19793,8710)) #8710
print('Our CoauthorCS',computeOur(18333,6805,163788))
print('Our Flickr',computeOur(89250,500,899756))
print('Our CoraFull',computeOur(19793,8710,126842))


print('-----------------------------')
print('AmazonCoBuyComputer',computeIA(13752,767))
print('AmazonCoBuyPhotoDataset',computeIA(7650,745))
print('WikiCSDataset',computeIA(11701,300))
print('Our AmazonCoBuyComputer',computeOur(13752,767,491722))
print('Our AmazonCoBuyPhotoDataset',computeOur(7650,745,238163))
print('Our WikiCSDataset',computeOur(11701,300,431726))

print('-----------------------------')
print('CoauthorPhysicsDataset',computeIA(34493,8415))
print('Reddit',computeIA(232965,602))
print('RomanEmpire',computeIA(22662,300))
print('Our CoauthorPhysicsDataset',computeOur(34493,8415,495924))
print('Our Reddit',computeOur(232965,602,114615892))
print('Our RomanEmpire',computeOur(22662,300,65854))

print('-----------------------------')
print('AmazonRatings',computeIA(24492,300))
print('MinesweeperDataset',computeIA(10000,7))
print('Questions',computeIA(48921,301))
print('Myket',computeIA(17988,33))
print('DBLP',computeIA(17716,1639))
print('DeezerEurope',computeIA(28281,128))

print('Our AmazonRatings',computeOur(24492,300,186100))
print('Our MinesweeperDataset',computeOur(10000,7,78804))
print('Our Questions',computeOur(48921,301,307080))
print('Our Myket',computeOur(17988,33,694121))
print('Our DBLP',computeOur(17716,1639,105734))
print('Our DeezerEurope',computeOur(28281,128,185504))



exit()
#
# x = np.arange(size)
# total_width, n = 0.7, 1
# width = total_width / n
# is_log=False
#
# fig, ax1 = plt.subplots(figsize=(5.5, 2.7), sharex=True)
# ax1.grid(True, axis='y', ls='dotted', zorder=100)
#
# # 柱状图
# bars3 = ax1.bar(x , y3, width=width, label='Other', color='white', zorder=10, edgecolor='black', linewidth='1',log=is_log)
# bars2 = ax1.bar(x , y2, width=width, label=r'Rot', color='silver', zorder=10, edgecolor='black', linewidth='1', hatch='//////',log=is_log)
# bars1 = ax1.bar(x , y1, width=width, label=r'MulC', color='silver', zorder=10, edgecolor='black', linewidth='1', hatch='xxxxxxxx',log=is_log)
#
# # 添加文本
# offset = 0.01  # 设置一个偏移量，用于文本的垂直位置
# for i,bar in enumerate(bars3):
#     height = bar.get_height()
#     ax1.text(bar.get_x() + bar.get_width() / 2.0+0.02, height + offset, f'{y1[i]/y3[i]*100:.2f}%', ha='center', va='bottom')
#
#
# ax1.set_ylabel('Percentage', fontsize=12)
# ax1.set_xticks(x)
# ax1.set_xticklabels(name,rotation=7,  fontsize=9) #
# ax1.legend(loc='center', bbox_to_anchor=(0.5, 1.2), ncol=7, fontsize=13)
#
# from matplotlib.ticker import FuncFormatter
#
# # 定义一个将小数转换为百分比形式的函数
# def to_percent(y, position):
#     # 将小数点形式的y值转换为百分比形式
#     return f"{100 * y:.0f}%"
#
# # 创建一个FuncFormatter对象，传入上面定义的转换函数
# percent_formatter = FuncFormatter(to_percent)
#
# # 使用FuncFormatter对象设置y轴的刻度格式
# ax1.yaxis.set_major_formatter(percent_formatter)
#
# plt.xlim((-n * width, size-0.3))  # 调整x轴的范围以适应所有的柱状图
# plt.ylim(0, 1.2)  # 可能需要调整以确保所有文本标签的可见性
# fig.tight_layout()
#
# plt.savefig('he_seg2.pdf')
# plt.show()