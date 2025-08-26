import numpy as np
import pandas as pd
from sympy import symbols, integrate, exp, Add
import math
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import quad
from sympy.polys.rings import _ring_cache
import psutil

import ctypes
ctypes.CDLL('libiomp5md.dll', mode=ctypes.RTLD_GLOBAL)  # Linux: libgomp.so
import time
import flow_submodel as fm
import gas_mass_balance as gsm
import cyc_submodel
import solid_balance_submodel as sb
import heat_transfer_submode as ht
import energy_submode_improved as es
import mode_coeffs

matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
"""
CFB模型主程序：

模块：流动模型、气体质量平衡、固体质量平衡（包括旋风分离器和返料装置）、传热模型、能量守恒以及汽水系统
1、划分小室，确定各小室几何参数、流动及燃烧情况
2、建立各小室气固质量平衡及能量守恒方程
3、每个时间步长下求解各模块
需清晰的逻辑包括：
1、流动模型为稳态，涉及到固体的需要注意宽筛分
2、由于平推假设，气体质量平衡方程于下流小室无关，可以沿风向依次求解，但是，固体有沉降项，需要所有控制体方程联立求解（能量守恒方程同）
3、流动模型为其他模块的基础，能量守恒模块建立在其他模块的基础之上
"""

# 常量确定
cp_p = 840  # 固体颗粒定压比热容，J/(kg·K)
rho_p = 1800  # 固体颗粒的密度，kg/m3（此项目未加脱硫剂，固体颗粒密度取1500而非2900）
rho_c = 1200  # 碳的密度
R = 8.314  # J⋅mol⁻¹K⁻¹，理想气体常数
g = 9.81  # 重力加速度，m/s2
dt = 1  # dt：时间步长，s
# 小室参数（小室的参数放到主程序中）
N = 37  # 划分的小室个数，根据实际情况调整
N_fur = 35  # 炉膛内小室个数（流动模型和气体质量平衡模块）
N_cyc = 26  # 与回料装置相通的再循环小室编号，根据实际情况调整（暂定为34，实际包含多个小室）
N_t = 8  # 含屏过的最后一个小室编号
N_vm = 35  # 含挥发分释放的小室个数
Num_cyc = 2 # 旋风分离器个数
D_bed = 2 * 5.28 * 10.27 / (5.28 + 10.27)  # 炉膛当量直径，m
A_bed = 5.28 * 10.27  # 炉膛截面积（长方形），m2
A_plate = 8.77 * 2.8  # 布风板横截面
A_cyc = 3.05 * 5.49  # 旋风分离器入口面积（运行操作规程）
H_bed = 33.1  # 炉膛高度，m（按操作规程需要改成32.2）
H_out = 29.8  # 炉膛出口高度（旋风分离器入口中心高度）
N_bed = 952  # 风板上的风帽数（运行操作规程）
H_con = 11.4  # 旋风分离器圆锥段
H_up = 4.4  # 旋风分离器直管段高度
D_up = 7.87  # 旋风分离器上口径
D_down = 1.34  # 旋风分离器下口径
A_coeffs, B_coeffs, C_coeffs, D_coeffs, E_coeffs, F_coeffs, G_coeffs, R_coeffs = \
    (mode_coeffs.A, mode_coeffs.B, mode_coeffs.C, mode_coeffs.D, mode_coeffs.E, mode_coeffs.F, mode_coeffs.G,
     mode_coeffs.R)  # 气体质量平衡方程系数（R为燃烧及化学反应项系数）
A_cab = mode_coeffs.A_cab  # 各小室横截面积，m2，二维，第二个维度第一个元素表示小室下界面面积，第二个元素表示小室上界面面积
A_w = mode_coeffs.A_w  # 各小室与水冷壁的传热面积，m²，划分好小室后即可明确（旋风分离器和返料装置中定为0，炉膛内的水冷壁如何考虑）
A_t = mode_coeffs.A_t  # 各小室与屏式换热器的传热面积，m²，划分好小室后即可明确（单屏：管排数（28），管径（30mm），管间距（60mm），屏数（6），有效高度（16m，根据小室拆开，沾污系数取0.9，角系数取0.9，烟气覆盖率取0.85）
h_cell = mode_coeffs.h_cell  # 初始化稀相区各小室的界面高度（考虑密相区的高度），划分好小室后即可明确
v_cabs = A_cab.mean(axis=1) * (h_cell[:, 0] - h_cell[:, 1])  # 各小室体积，m3
v_cyc = np.pi * (D_up / 2) ** 2 * H_up + np.pi / 12 * (D_up ** 2 + D_up * D_down + D_down ** 2) * H_con  # 旋风分离器体积
v_ls = D_down ** 2 / 4 * np.pi * (20.5-8.3)  # 回料装置体积
h_cab_den = 0.3  # 密相床区域单个小室高度，m（根据实际情况调整）
phi = 0.85       # 颗粒球形度
P0 = 101325      # 炉膛出口压力，Pa（参考压力）
T_ref = 273.15   # 标况温度，K
dp = np.array([0.06, 0.09, 0.125, 0.16, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0, 2.0,
               3.0]) * 0.001  # 宽筛分颗粒粒径，mm->m[0.06, 0.09, 0.125, 0.16, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0, 2.0, 3.0, 6.0, 10]考虑破碎和磨蚀进行调整
X0 = np.array([0.1, 0.226, 0.027, 0.027, 0.03, 0.09, 0.09, 0.25, 0.06, 0.05, 0.025,
               0.025])  # 添加灰各个粒径下颗粒组分，同时作为组分初始条件，注意为颗粒碎裂以后的分布[0.04, 0.026, 0.027, 0.027, 0.03, 0.025, 0.025, 0.06, 0.05, 0.25, 0.09, 0.09, 0.2, 0.06]

# 输入参数的确定（ Q_air：各小室给风流率，m3/s，需转化为mol/s（mol/s=P×(m³/s)/(R×T)）
# W_air：给风流率，kg/s，注意与Q_air 区分（Q_air不考虑副床，直接将给风作为炉膛小室的边界条件，而W_air则按副床有给风，考虑副床小室的能量守恒），T_air：给风温度，K，W_fa：给煤（固体）流率，kg/s，T_fa：给煤温度，K
# Q1_air：一次风流量，Q2_air：二次风流量，Q3_air：返料风流量，W_fa：给煤（固体）流率，kg/s
# data = pd.read_excel("input.xlsx", header=None, skiprows=1)  # 默认读取第一个 Sheet
# W_fa0 = data.iloc[:, 5] / 3.6  # 读取第 6 列（所有行），kg/s
cal_len = 10000
# 设计工况下参数
W_fa0 = np.full(cal_len, 10.0)
dTNh_s = P0 / (T_ref * R * 3600)  # Nm3/h转化为mol/s（实际工况的温度和压力与标况不同，实际体积流量会变化，但摩尔流量不变）
# Q1_air = data.iloc[:, 6] * dTNh_s
# Q2_air = data.iloc[:, 7] * dTNh_s
# Q3_air = data.iloc[:, 8] * dTNh_s
# dP_fur = data.iloc[:, 9] + data.iloc[:, 10] * 10 ** 3  # 炉膛压差（料层+炉膛），Pa
Q3_air = np.full(cal_len, 7140) * dTNh_s
Q1_air = (238000 * P0 / ((T_ref+20) * R * 3600)-Q3_air)/2
Q2_air = (238000 * P0 / ((T_ref+20) * R * 3600)-Q3_air)/2
dP_fur = np.full(cal_len, 10000)

T0_air = T_ref + 20  # 空预进风温度，K（设计参数）
T1_air = np.ones((Q1_air.shape[0])) * 400  # 初始化一个一次风温度，K（后续根据点位数据读取）
T2_air = np.ones((Q1_air.shape[0])) * 450  # 初始化一个二次风温度，K（后续根据点位数据读取）
T3_air = np.ones((Q1_air.shape[0])) * T0_air  # 初始化一个返料风温度，K（后续根据点位数据读取）
T_fa0 = np.ones((Q1_air.shape[0])) * 300  # 初始化给料温度，K（后续根据点位数据读取）

# X_VM = W_fa0 * 0.2204 * 1000      # 各小室挥发分（0.2204为20240401当天煤中挥发分含量，挥发分单位：g/s）
X_VM = 0.293
M_S = W_fa0 * 0.008 * 1000 / 32  # 煤中硫元素的量（0.0042为20240401当天煤中硫含量,单位：mol/s）
n_Mass = W_fa0 * 0.00443 * 1000 / 14  # 煤中氮化合物量，mol/s（根据设计参数，煤中氮的含量为4.43%，氮摩尔质量为14，根据入煤报表，硫元素含量与设计参数差别较大，结合硫元素修改下）
# 将导入的一二次风及返料风合并到给风中
Q_air = np.zeros((Q1_air.shape[0], N - 2))  # 复核及优化（气体质量平衡，只考虑炉膛内的小室）
Q_air[:, -1] = Q1_air  # 复核及优化
Q_air[:, 22] = Q2_air  # 复核及优化
Q_air[:, 23] = Q3_air  # 复核及优化
W_air = np.zeros((Q1_air.shape[0], N))      # 复核及优化（能量守恒，需要考虑旋风分离器和返料装置）
W_air[:, -1] = Q1_air  # 复核及优化
W_air[:, 24] = Q2_air  # 复核及优化
W_air[:, 25] = Q3_air
W_air[:, 0] = Q3_air  # 复核及优化
T_air = np.zeros((Q1_air.shape[0], N))      # 复核及优化（能量守恒，需要考虑旋风分离器和返料装置）
T_air[:, -1] = T1_air
T_air[:, 24] = T2_air
T_air[:, 0] = T3_air

# 给煤
W_fa = np.zeros((Q1_air.shape[0], N))  # 复核及优化（固体质量平衡，只考虑炉膛内的小室，在能量守恒中，需要考虑旋风分离器和返料装置）
W_fa[:, 24] = W_fa0
G_fa = np.zeros((N_fur, len(dp)))  # 初始化给料速率，也是二维矩阵，0维度为小室，1维度为颗粒档
T_fa = np.zeros((Q1_air.shape[0], N))
T_fa[:, 24] = T_fa0

X_fa_km = np.zeros((N_fur, len(X0)))  # 初始化给料中各颗粒档固体的质量百分数,已知条件
X_fa_km[:] = X0  # 初始化
Xfac_km = np.zeros((N_fur, len(dp)))  # 初始化给料中碳的质量百分数,已知条件
Xfac_km[22, :] = 0.5221  # 给料小室输入给煤中碳的质量百分数
Xfaca_km = np.zeros((N_fur, len(dp)))  # 给料中氧化钙的质量百分数,已知条件（高阳项目未添加脱硫剂）
Xca_km = np.zeros((N, len(dp)))  # 初始化各小室中氧化钙的质量百分数（高阳项目未添加脱硫剂）
# 参数初始化
# 流动模型计算需要初始化的一些参数（G_hk_up~e_k_ksf）
G_hk_up = np.zeros((len(h_cell), len(dp)))  # K档颗粒稀相区各小室上界面的质量流率，kg/s
G_hk_down = np.zeros((len(h_cell), len(dp)))  # K档颗粒稀相区各小室下界面的质量流率，kg/s
e_k = np.zeros((len(h_cell), len(dp)))  # 各小室内的平均孔隙率（K档颗粒）
e_k_ksf = np.zeros(len(h_cell))  # 各小室内的平均孔隙率
h_den = 3.9  # 初始化密相床高度
# carbon_content = np.zeros((N, len(dp)))     # 由于气体模块先于固体模块计算，因此需要初始化一个各小室含碳量
carbon_content = np.full((N, len(dp)), 0.02)  # 初始化每个小室含碳量为1%~3%
n_mass = np.zeros(N_fur)  # 初始化各小室氮氧化物含量
m_s = np.zeros(N_fur)  # 初始化各小室硫含量
x_vm = np.zeros(N_fur)  # 初始化各小室挥发分含量（g/s）
Xc_km0 = carbon_content  # 初始化碳和氧化钙的质量百分数（固体模块计算需要）
Xca_km0 = np.zeros((N, len(dp)))
M_cyck0 = np.zeros(len(dp))  # 初始化上一时刻旋风分离器k档颗粒滞留量，初始化上一时刻返料装置k档颗粒滞留量
M_lsk0 = np.zeros(len(dp))
W_rek = np.zeros((N_fur, len(dp)))  # 初始化返料装置返料量
Gd = np.zeros((N_fur, len(dp)))  # 初始化各小室排渣率，后续将流动模型计算出来的排渣填充到相应小室中去，kg/s
Vin = 16 * A_cyc  # 由于流动模型先于气体模块计算，因此需要初始化一个旋风分离器入口气体体积流率，m3/s（入口烟气速度*入口面积）
#T = np.full(N, 900 + T_ref, dtype=np.float64)  # 初始化各小室温度，K，后续需要通过能量方程的求解得到各小室温度（求解能量守恒方程后更新）
T = np.linspace(900+T_ref, 850+T_ref, N)  # 从900递减到850，均匀分布
T0 = np.copy(T)  # 初始化上一时刻各小室温度，K
T_w = np.full(N, 308 + T_ref, dtype=np.float64)  # 初始化水冷壁壁温，K
T_secondary_w = np.full(N, 308 + T_ref, dtype=np.float64)  # 初始化副床水冷壁壁温，K
T_t = np.full(N, 419 + T_ref)  # 初始化屏式换热器壁温，K
T_cor, T_ann, T_den, T_secondary_den = np.copy(T), np.copy(T), np.copy(T), np.copy(
    T)  # 均可用小室的温度序列来表示（此次建模不区分核心区和环形区温度，密相床的温度包含在小室温度序列内，可优化）T_cor: 核心区温度 [K]，T_ann: 环形区温度 [K]，T_den：密相床温度，T_secondary_den：副床床温
T_cyc_w = 308 + T_ref  # 旋风分离器处壁温，K （需复核）
P = np.full(N_fur, 101325) # 各小室压力，Pa
dP = np.zeros(N_fur)
# 1. 用均匀分布随机初始化（范围 [0, 1)）
y_gas_0 = np.array([0.05, 0.01, 0.15, 0.001, 0.0004, 0.7386, 0.05])  # [f_o2, f_co, f_co2, f_so2, f_no, f_n2, f_h2o]
Y_gas_0 = np.tile(y_gas_0, (N_fur, 1))  # 初始化上一时刻每个小室的各气体体积浓度，m3/m3
# 2. 对每一行进行归一化（使行和为1）
G_gas_0 = np.ones(N_fur) * 2756  # 初始化上一时刻每个小室的气体流率，mol/s（按照炉膛内烟气截面流速4.7m/s，结合炉膛截面，得到烟气流率254m3/s即2756mol/s）
M_cyc0 = 2580/Num_cyc     # np.pi / 12 * (D_up ** 2 + D_up * D_down + D_down ** 2) * H_con  # 初始化旋风分离器中的颗粒滞留量，灰斗容积*10kg/m3=2216）
M_ls0 = 35668/Num_cyc   # v_ls * rho_p * (1 - 0.5) * g  # 初始化返料装置中的颗粒滞留量，kg（固体滞留量：循环倍率*平均停留时间/返料效率）
dp_ls0 = 200  # 初始化返料装置压降，Pa（一般比炉膛压降稍大）
W_re0 = W_fa0[0]*20/Num_cyc  # 初始化返料装置总返料量
# 时间序列输出参数（待观测变量）
total_G_outk = np.zeros(Q1_air.shape[0])  # 初始化炉膛出口固体夹带量，kg/s
total_heat_w = np.zeros((Q1_air.shape[0], N_fur))  # 初始化各小室传热系数，W/(m2 k)
total_h_den = np.zeros(Q1_air.shape[0])  # 初始化密相床高时间序列，m
total_M_g = np.zeros((Q1_air.shape[0], N_fur))  # 初始化各小室气体总摩尔数时间序列，mol
total_M_p = np.zeros((Q1_air.shape[0], N_fur))  # 初始化各小室固体总质量时间序列，kg
total_X_km = np.zeros((Q1_air.shape[0], len(dp)))  # 初始化密相区宽筛分质量份额时间序列
total_Y_o2 = np.zeros((Q1_air.shape[0], N_fur))  # 初始化每个小室各类气体体积浓度时间序列
total_Y_co = np.zeros((Q1_air.shape[0], N_fur))
total_Y_co2 = np.zeros((Q1_air.shape[0], N_fur))
total_Y_so2 = np.zeros((Q1_air.shape[0], N_fur))
total_Y_no = np.zeros((Q1_air.shape[0], N_fur))
total_Y_n2 = np.zeros((Q1_air.shape[0], N_fur))
total_Y_h2o = np.zeros((Q1_air.shape[0], N_fur))
total_W_g = np.zeros((Q1_air.shape[0], N_fur))  # 初始化每个小室气体质量流率时间序列，维度大小为[N_t,N]（N_t表示时间序列长度）
total_e_k = np.zeros((Q1_air.shape[0], N_fur, len(dp)))  # 初始化各小室各颗粒档孔隙率时间序列
total_M_cyc = np.zeros(Q1_air.shape[0])  # 初始化各时刻旋风分离器内颗粒滞留量，kg/s
total_M_ls = np.zeros(Q1_air.shape[0])  # 初始化各时刻返料装置内颗粒滞留量，kg/s
R_cyc = np.zeros(Q1_air.shape[0])
total_T = np.zeros((Q1_air.shape[0], N))  # 初始化每个小室温度时间序列
total_P = np.zeros((Q1_air.shape[0], N_fur))  # 初始化每个小室压力时间序列
Q_heat_w = Q_heat_t = np.zeros((Q1_air.shape[0], N))  # 初始化燃烧系统向水冷壁和屏式换热器传热量，W

# 各模块类实例化
flow = fm.Flow(dp, phi, rho_p, rho_c, D_bed, A_bed, A_plate, H_bed, H_out, N_bed, X0, A_cyc)
gas = gsm.GAS(N_fur, dt, dp, rho_p, rho_c, N_vm, H_bed)
cyc = cyc_submodel.cyc_mass_energy()
solid = sb.solid_mass(N_fur, v_cabs, rho_p, dt)
hr = ht.Heat_trans(cp_p,rho_p,dp,N,N_t)
eng = es.energy(N, N_cyc, cp_p)


# 物性更新（需要更新的参数较多，考虑单独建一个物性更新库实时导入，根据此函数进行扩展）
def update_gas_properties(T, P):
    """
    更新气体物性(温度/压力影响)
    参数:
    T - 温度(K)
    P - 压力(Pa)
    """
    # 理想气体状态方程 + Sutherland粘度公式
    rho_g = 0.316  # 气体密度，850℃，1个标注大气压，若要考虑起炉过程，可以写成拟合公式
    mu_g = 4.5e-5  # 气体动力粘度(Pa·s)
    cp_g = 34.56  # 气体摩尔定压比热，J/(mol·K)
    thermo_g = 0.25  # 密相区气体导热系数，0.15~0.35W/(m⋅K)
    K_g = 0.06  # 气膜导热系数，受到气体种类、温度、压力以及颗粒浓度等多种因素的影响，通常在0.02−0.1 W/(m⋅K)
    return rho_g, mu_g, cp_g, thermo_g, K_g


# 创建两个画布
# fig1, ax1 = plt.subplots(figsize=(8, 4))  # 画布1，新建一个画布（在循环外创建，否则每次循环均会创建一个画布）气体浓度
# fig2, ax2 = plt.subplots(figsize=(8, 4))  # 画布2，换热系数
# fig3, ax3 = plt.subplots(figsize=(8, 4))  # 画布3，固体体积浓度
# 初始化图形
plt.ion()  # 开启交互模式
# 创建1行2列的子图布局
fig, axs = plt.subplots(2, 2, figsize=(12, 4))  # figsize控制总画布大小
ax1, ax2, ax3, ax4 = axs.ravel()  # 展平为一维数组后解包

time_data, state_data1 = [], []  # 存储时间和状态量
line1, = ax1.plot([], [], 'b-', label='炉膛出口氧气浓度%')  # 初始空曲线
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('炉膛出口氧气浓度')
ax1.set_title('炉膛出口氧气浓度时间序列')
ax1.legend()
ax1.grid(True)

state_data2 = []  # 存储时间和状态量
line2, = ax2.plot([], [], 'r-', label='循环倍率')  # 初始空曲线
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('循环倍率')
ax2.set_title('循环倍率')
ax2.legend()
ax2.grid(True)

state_data3 = []  # 存储时间和状态量
line3, = ax3.plot([], [], 'r-', label='炉膛出口温度')  # 初始空曲线
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('炉膛出口温度（℃）')
ax3.set_title('炉膛出口温度')
ax3.legend()
ax3.grid(True)

state_data4 = []  # 存储时间和状态量
line4, = ax4.plot([], [], 'r-', label='旋风分离器内固体滞留量')  # 初始空曲线
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('旋风分离器内固体滞留量（kg）')
ax4.set_title('旋风分离器内固体滞留量')
ax4.legend()
ax4.grid(True)

start_time = time.time()  # 记录开始时间
for i in range(cal_len):  # Q_air(时间维度，N)为一个二维向量，其第一个维度是时间序列，时间序列的间隔即为时间步长，也可以取其他参数的时间序列维度
    print("i", i)
    # 物性更新
    [rho_g, mu_g, cp_g, thermo_g_mode, K_g_mode] = update_gas_properties(T[2:], P)
    rho_g_flow = rho_g  # 流动模型传入的物性参数为各小室平均值（维度大小问题）
    mu_g_flow = mu_g
    thermo_g = thermo_g_mode
    K_g = K_g_mode

    # u0 = (Q1_air[i]+Q2_air[i]+Q3_air[i])*R*T[2]/(P[0]*A_bed)  # U0为一个时间序列参数，取i时刻的表观速度，m/s
    # u0_den = (Q1_air[i]*R*T[-1]/(P[-1]*A_plate*0.1))
    u0 = (Q1_air[i]+Q2_air[i]+Q3_air[i])*R*(T_ref+850)/(P0*A_bed)  # U0为一个时间序列参数，取i时刻的表观速度，m/s（表观风速工况温度和压力如何选取）
    # u0_den = (Q1_air[i]*R*T_ref/(P0*A_plate*0.025))
    u0_den = np.copy(u0)
    print('u0', u0)
    # print('u0_den', u0_den)

    def calculate_flow_parameters(h_cell, n_den):
        ep_den_ksf = np.sum((1 - e_denk) * X_km)  # 密相区宽筛分的颗粒体积浓度
        Wdrk = Wdr * X_km

        for i in range(len(h_cell)):  # 计算各小室流动参数,若有两个维度，第一维度是小室数，第二个维度是颗粒分档
            # h = symbols('h')
            # 清空缓存（防止内存爆炸）
            if psutil.virtual_memory().percent > 80:  # 内存使用超过80%时清空
                _ring_cache.clear()

            if i < len(h_cell) - n_den:  # 稀相区小室个数
                def integrand(h_val, j):
                    return e_TDHk[j] + (e_denk[j] - e_TDHk[j]) * np.exp(-ak[j] * h_val)
                for j in range(len(dp)):
                    # e_hkf_value = e_TDHk[j] + (e_denk[j] - e_TDHk[j]) * exp(-ak[j] * h)  # h高度上k档颗粒稀相区小室内的孔隙率计算函数
                    # e_k[i, j] = integrate(e_hkf_value, (h, h_cell[i, 1] - h_den, h_cell[i, 0] - h_den)) / (
                    #         h_cell[i, 0] - h_cell[i, 1])  # h高度上k档颗粒稀相区小室内的平均孔隙率
                    result = quad(lambda h: integrand(h, j), h_cell[i, 1] - h_den, h_cell[i, 0] - h_den)
                    integral_result = result[0]  # 只取积分结果部分
                    e_k[i, j] = integral_result / (h_cell[i, 0] - h_cell[i, 1])
                e_k_ksf[i] = np.sum(e_k[i, :] * X_km)
                dP[i] = g * rho_p * (1 - e_k_ksf[i]) * (h_cell[i, 0] - h_cell[i, 1])  # 前i个小室压降累和
                P[i] = P0 + np.sum(dP[:i])
                G_hk_up[i, :] = (G_TDHk + (G_densk - G_TDHk) * np.exp(  # 以下两个公式注意考虑宽筛分（注意流动模型计算出来的流率是单位面积的，需要乘上小室截面面积）
                    -ak * (h_cell[i, 0] - h_den))) * X_km * A_cab[
                                    i, 1]  # K档颗粒稀相区各小室上界面的质量流率，kg/s,h_cell 是稀相区各小室的界面高度，考虑密相区的高度
                G_hk_down[i, :] = (G_TDHk + (G_densk - G_TDHk) * np.exp(
                    -ak * (h_cell[i, 1] - h_den))) * X_km * A_cab[i, 0]  # K档颗粒稀相区各小室下界面的质量流率
            else:
                dP[i] = g * rho_p * ep_den_ksf * (h_cell[i, 0] - h_cell[i, 1])  # 前i个小室压降累和
                P[i] = P0 + np.sum(dP[:i])
                e_k[i, :] = e_denk
                e_k_ksf[i] = np.sum(e_denk * X_km)
                G_hk_up[i, :] = G_densk * X_km * A_cab[i, 1]
                G_hk_down[i, :] = G_densk * X_km * A_cab[i, 0]
        G_ksf_dowm = np.sum(G_hk_down, axis=1)
        G_ksf_up = np.sum(G_hk_up, axis=1)  # 将其每列（颗粒档）进行加和

        return P, dP, e_k, e_k_ksf, G_hk_up, G_hk_down, Wdrk, G_ksf_up, G_ksf_dowm  # 流动模型输出的质量流率均为kg/s（已经考虑相应截面积）


    if i == 0:  # 模块功能：赋初值
        # 调用流动模型（注意带k的均为宽筛分下情况）
        X_km, h_den, G_denk, G_Hk, effk_cyc, eff_bed_out, G_TDHk, G_densk, e_TDHk, e_denk, ak, Wdr, U_mfk, e_mfk, u_tk, G_outk = flow.solve_flow_submodel(
            u0, u0_den, Vin, rho_g_flow, mu_g_flow, h_den, 0.25*W_fa0[i], dP_fur[i])
        n_den = int(h_den // h_cab_den)  # 密相床小室个数（向下取整，过渡区小室个数也会更新）
        [P, dP, e_k, e_k_ksf, G_hk_up, G_hk_down, Wdrk, G_ksf_up, G_ksf_down] = calculate_flow_parameters(h_cell, n_den)
        M_g_0 = P * e_k_ksf * v_cabs / (R * T[2:])  # M_g：当前时刻各小室气体总摩尔数，mol
        total_h_den[i] = h_den  # 储存当前时刻密相区床高，m
        total_M_g[i, :] = M_g_0  # 储存当前时刻各小气体总摩尔数，mol
        M_p_0 = rho_p * v_cabs * (1 - e_k_ksf)  # M_p：当前时刻各小室固体总质量，kg（注意固体质量平衡方程中有各小室各档颗粒的固体质量计算，后续可以优化一下）
        total_M_p[i, :] = M_p_0  # 储存当前时刻各小室固体总质量，kg
        X_km_0 = np.copy(X_km)
        total_X_km[i] = X_km  # 储存当前时刻密相区颗粒宽筛分质量份额
        e_k0 = np.copy(e_k)
        total_e_k[i, :, :] = e_k
        total_P[i, :] = P
        total_G_outk[i] = np.sum(G_outk)  # 储存当前时刻炉膛出口固体夹带量，kg/s
        total_M_cyc[i] = M_cyc0
        total_M_ls[i] = M_ls0
        X_cyc0 = np.copy(X_km)  # 初始化旋风分离器宽筛分系数
        X_ls0 = np.copy(X_km)  # 初始化返料装置宽筛分系数
        continue

    # 调用流动模型（注意带k的均为宽筛分下情况）
    X_km, h_den, G_denk, G_Hk, effk_cyc, eff_bed_out, G_TDHk, G_densk, e_TDHk, e_denk, ak, Wdr, U_mfk, e_mfk, u_tk, G_outk = flow.solve_flow_submodel(
        u0, u0_den, Vin, rho_g_flow, mu_g_flow, h_den, 0.25*W_fa0[i], dP_fur[i])
    """
            流动模型

            参数:
            u0 - 表观风速，m/s
            Vin - 进入旋风分离器气体体积流量，m3/s
            rho_g_flow - 气体密度，kg/m3（流动模型传入的物性参数为各小室平均值）
            mu_g_flow - 气体动力粘度（pa*s)或kg/(m.s)
            h_den - 初始化的密相床高度，将上一时刻计算得到的密相床高作为下一次的初始值，减少迭代次数
            W_fa - 给料（灰）质量，kg/s（注意流动模型的输入不区分小室，为实际给料量）

            返回:
            X_km - 密相区固体质量份额
            h_den - 密相区床高，m
            G_denk - 密相床内部各档颗粒的质量流率，kg/s
            G_Hk - 炉膛出口的质量流率,kg/s 
            Umf - 最小流化速度(m/s)
            e_mfk- 最小流化孔隙率
            effk_cyc - 旋风分离器的分离效率
            eff_bed_out - 主床出口的分离效率
            G_TDHk - TDH高度以上的饱和夹带率，kg/m2/s
            G_densk - 密相区表面扬析率，kg/m2/s
            e_TDHk - TDH高度以上的孔隙率
            e_denk - 密相区孔隙率（密相区孔隙率：0.5~0.8，设计时取0.6~0.75）
            ak - 衰减系数
            Wdr - 排渣量，kg/s
            U_mfk - 最小流化速度，m/s
            u_tk - 颗粒终速度，m/s
            G_outk - 进入旋风分离器的K档颗粒流率，kg/s（各颗粒档总夹带量应在50~200kg/s范围内）
            """
    # 根据流动模型计算结果得到各小室流动情况（需要根据流动模型计算出的密相床高度讨论密相床小室个数，注意：密相床区域更新后，相应的参数也要跟着调整）（小室从炉膛最高处开始，流动模型只考虑炉膛）
    total_h_den[i] = h_den
    total_G_outk[i] = np.sum(G_outk)  # 储存当前时刻炉膛出口固体夹带量，kg/s
    n_den = math.ceil(h_den / h_cab_den)  # 密相床小室个数（向上取整，过渡区小室个数也会更新）
    N_dil = N - n_den  # 稀相区最后一个小室编号，用于更新传热模型输入
    [P, dP, e_k, e_k_ksf, G_hk_up, G_hk_down, Wdrk, G_ksf_up, G_ksf_down] = calculate_flow_parameters(h_cell, n_den)
    """
            流动模型（与小室相对应）

            参数:
            h_cell - 稀相区各小室的界面高度，考虑密相区的高度
            n_den - 密相床小室个数（根据密相区床高调整）

            返回:
            P - 各小室压力，Pa
            dP - 各小室压降，Pa
            e_k - 各小室内的平均孔隙率（K档颗粒）
            e_k_ksf - 各小室内的平均孔隙率
            G_hk_up - K档颗粒稀相区各小室上界面的质量流率，kg/s
            G_hk_down - K档颗粒稀相区各小室下界面的质量流率，kg/s
            Wdrk - K档颗排渣量，kg/s
            G_ksf_up - 稀相区各小室上界面的质量流率，kg/s
            G_ksf_dowm - 稀相区各小室下界面的质量流率，kg/s
            """
    print("密相床床高：", h_den)
    print("各小室颗粒体积分数：", 1 - e_k_ksf)
    print("密相床各档颗粒质量份额：", X_km)
    print("沿炉膛高度压降（Pa）：",P-P0)
    epsilon_p_cl = 1 - e_k_ksf  # 颗粒团的固体体积份额（与环形区内的固体体积份额相等），epsilon_p_c：核心区固体体积份额
    epsilon_p_c = 1 - e_k_ksf
    M_g = P * e_k_ksf * v_cabs / (R * T[2:])  # M_g：当前时刻各小室气体总摩尔数，mol
    M_p = rho_p * v_cabs * (1 - e_k_ksf)  # M_p：当前时刻各小室固体总质量，kg（注意固体质量平衡方程中有各小室各档颗粒的固体质量计算，后续可以优化一下）
    total_M_g[i, :] = M_g  # 储存当前时刻各小气体总摩尔数，mol
    total_M_p[i, :] = M_p  # 储存当前时刻各小室固体总质量，kg
    total_X_km[i] = X_km  # 储存当前时刻密相区颗粒宽筛分质量份额
    total_e_k[i, :, :] = e_k
    total_P[i, :] = P  # 储存当前时刻各小室压力，Pa
    M_g_0 = total_M_g[i - 1, :]  # 上一时刻各小室气体总摩尔数赋值
    M_p_0 = total_M_p[i - 1, :]  # 上一时刻各小室固体总质量赋值
    X_km0 = total_X_km[i - 1]  # 上一时刻密相区宽筛分质量份额赋值
    e_k0 = total_e_k[i - 1, :]  # 上一时刻
    W_p = np.copy(G_ksf_up)  # 当前时刻各小室固体流率，kg/s（区分G_ksf_up和G_hk_down（小室上下界面问题））
    n_mass[-N_vm:] = n_Mass[i] / N_vm  # 当前时刻各小室氮化合物含量，mol/s
    m_s[-N_vm:] = M_S[i] / 19  # 当前时刻各小室硫含量，mol/s（煤中硫在密相区及过渡区释放）
    x_vm[-N_vm:] = X_VM  # 当前时刻各小室挥发分含量，g/s
    M_km = np.abs((1 - e_k)) * X_km * rho_p * v_cabs[:, None]  # 当前时刻各小室灰质量
    M_km0 = np.abs((1 - e_k0)) * X_km0 * rho_p * v_cabs[:, None]  # 上一时刻各小室灰质量
    # 所有小室的固体质量分数
    # X_km_cabs = M_km / np.sum(M_km, axis=1, keepdims=True)  # keepdims=True 保持求和后的维度为 (35, 1) ，第一个维度是小室，第二个维度是颗粒（稀相区各档颗粒质量份额与密相区相差较大，计算时需要考虑用哪个质量份额）
    ## 调用气体质量平衡模型，得到各小室气体流率（流入流出G_gas）、各小室气体体积浓度、及固体其他模块计算需要的参数（气体暂时只考虑炉膛内）
    # result_gas = G_gas, heat_cabs, Y_o2, Y_co, Y_co2, Y_so2, Y_no, Y_n2, Y_h2o, Y_gas_0,G_gas_0, R_ch ar, R_ca
    result_gas = gas.total_gas_mass_balance(M_g_0, M_g, Y_gas_0, G_gas_0, P, e_k, e_k_ksf, e_mfk, X_km, u_tk, carbon_content[2:, :],
                                            Xca_km[2:, :], T[2:], x_vm, m_s, n_mass, W_fa0[i], Q_air[i,:], v_cabs, mu_g, rho_g,
                                            A_coeffs, B_coeffs, C_coeffs, D_coeffs, E_coeffs, F_coeffs, G_coeffs,
                                            R_coeffs, h_cell)
    """
                气体质量平衡

                参数:
                M_g_0 - 上一时刻各小室气体总摩尔数，mol
                M_g - 当前时刻各小室气体总摩尔数，mol（流动模型计算）
                Y_gas_0 - 上一时刻各小室各气体体积浓度，m3/m3（具体形式：Y_gas=[Y_o2,Y_co,Y_co2,Y_so2,Y_no,Y_n2,Y_h2o]）
                G_gas_0 - 上一时刻各小室气体摩尔流率，mol/s
                P - 各小室压力，Pa（流动模型计算）
                e_k - 各小室内的平均孔隙率（K档颗粒）
                e_k_ksf - 各小室内的平均孔隙率
                e_mfk - 最小流化孔隙率
                X_km_cabs - 各小室各档颗粒质量分数
                u_tk - 各档颗粒终端速度，m/s
                carbon_content - 各小室含碳量（固体模块计算）
                Xca_km - 各小室氧化钙质量份额
                T - 各小室温度，K（能量守恒计算）
                x_vm - 各小室煤中挥发分含量（单位：100%）
                M_S - 煤中硫元素的量（单位：mol/s？）
                n_Mass - 煤中氮化合物量，mol/s
                W_fa0[i] - 当前时刻给煤量，kg/s
                Q_air - 各小室给风流率，m3/s，需转化为mol/s（mol/s=P×(m³/s)/(R×T)）
                v_cabs - 各小室体积，m3
                mu_g - 气体动力粘度，Pa s
                rho_g - 气体密度
                A_coeffs - 元素依次表示是否含焦炭燃烧消耗的氧气、挥发分燃烧消耗的氧气、CO燃烧消耗的氧气、给风带入的氧气、吸收SO2消耗的氧气以及生成SO2消耗的氧气
                B_coeffs - 元素依次表示是否含CO燃烧消耗的CO、CO2还原生成的CO、焦炭燃烧生成的CO、NO还原反应消耗的CO
                C_coeffs - 元素依次表示是否含CO燃烧生成的CO2、挥发分燃烧生成的CO2、焦炭燃烧生成的CO2、CO2还原反应消耗的CO2、碳酸钙分解产生的CO2、NO还原反应生成的CO2
                D_coeffs - 元素依次表示是否含煤中硫氧化生成的SO2、GaO吸收的SO2
                E_coeffs - 元素依次表示是否含煤中氮氧化物转化的NO、NO还原反应消耗的NO
                F_coeffs - 元素依次表示是否含NO还原反应生成的N2、给风带入的N2
                G_coeffs - 元素依次表示是否含挥发分燃烧生成的水蒸气和挥发出来的水蒸气
                R_coeffs - 元素依次表示是否含挥发分燃烧、焦炭燃烧、SO2的吸收以及NO的还原
                h_cell - 各小室上下界面高度，m

                返回:
                W_g - 当前时刻各小室气体摩尔流率，mol/s
                heat_cabs - 各小室（燃烧）化学反应产生的热量，W（能量守恒模块输入）
                Y_o2, Y_co, Y_co2, Y_so2, Y_no, Y_n2, Y_h2o - 当前时刻各小室各气体（7种气体）的气体浓度 
                Y_gas_0 - 上一时刻各小室各气体体积浓度，m3/m3
                G_gas_0 - 上一时刻各小室气体摩尔流率，mol/s
                R_char - 各小室焦炭燃烧反应速率，kg/s，维度大小为[N,len(dp)]（固体模块输入）
                R_ca - 各小室氧化钙燃烧反应速率，kg/s（固体模块输入）
                """
    W_g, heat_cabs, Y_gas_0, G_gas_0, R_char, R_ca = (result_gas[0], result_gas[1], result_gas[9],
                                                      result_gas[10], result_gas[11], result_gas[-1])
    heat_gas = np.sum(heat_cabs)  # 炉膛内因燃烧/化学反应产生的热量
    R_chark = (np.sum(R_char, axis=1)) * 1000  # 各小室焦炭燃烧反应速率，g/s
    print("单位高度上焦炭反应速率：", R_chark/(h_cell[:,0]-h_cell[:,1]))
    V_g = W_g * R * T[2:] / P  # 气体体积流率,m3/s（流动模型的输入）
    Vin = V_g[0] / Num_cyc  # 注意流动模型需要输入的是进入旋风分离器的气体体积流率（旋风分离器有两个）
    u_in = Vin / A_cyc  # 烟气质量守恒输入气流速度（旋风分离器入口气流速度），m/s
    # 储存各个时刻每个小室各类气体体积浓度，m3/m3（输出观测量）
    (total_Y_o2[i, :], total_Y_co[i, :], total_Y_co2[i, :], total_Y_so2[i, :], total_Y_no[i, :], total_Y_n2[i, :],
     total_Y_h2o[i, :]) = \
        (result_gas[2], result_gas[3], result_gas[4], result_gas[5], result_gas[6], result_gas[7], result_gas[8])
    total_W_g[i, :] = W_g  # 储存各个时刻每个小室的气体流率，mol/s
    print("炉膛内氧气浓度分布：", result_gas[2])
    print("炉膛内CO浓度分布：", result_gas[3])
    print("炉膛内CO2浓度分布：", result_gas[4])
    print("炉膛内SO2浓度分布：", result_gas[5])
    print("炉膛内NO浓度分布：", result_gas[6])
    print("炉膛内N2浓度分布：", result_gas[7])
    print("炉膛内H2O浓度分布：", result_gas[8])
    print("气体流率：mol/s", result_gas[0])

    ## 调用固体质量平衡模型（包括炉膛和旋风分离器及返料装置），各小室固体流率、各小室固体返混流量、脱硫剂和碳份额（假设烟气都通过旋风分离器出口进入尾部烟道）
    [M_cyc, X_cyc, dp_cyc, M_ls, dp_ls, W_re, X_ls, w_rek, G_downk, G_flyK, M_cyck, M_lsk,
     dp_sp] = cyc.calculate_cyc_parameters(u_in, u_tk,
                                           G_outk/Num_cyc, effk_cyc, M_cyc0, X_cyc0, X_km, dP_fur[i], M_ls0, dp_ls0, W_re0,
                                           X_ls0, np.sum(e_mfk*X_km), u0)
    """
                旋风分离器和返料装置

                参数:
                u_in - 烟气质量守恒输入气流速度，m/s
                u_tk - k档终端速度，m/s，流动模型输出
                G_outk - k档离开炉膛固体颗粒流率，kg/s，流动模型输出
                effk_cyc - 旋风分离器的分离效率，流动模型输出，避免重复计算
                M_cyc0 - 旋风分离器中的初始颗粒滞留量
                X_cyc0 - 旋风分离器中的初始颗粒质量分数
                X_km - 当前时刻炉膛密相区的宽筛分质量份额，作为旋风分离器宽筛分质量份额的初始值，流动模型输出
                dp_fur - 炉膛压降，Pa,床压，主程序输入
                M_ls0 - 返料装置中的初始颗粒滞留量，kg，初始值都用上一时刻的计算值
                dp_ls0 - 返料装置压降迭代计算的初始值
                W_re0 - 返料装置总返料量迭代计算的初始值
                X_ls0 - 返料装置宽筛分系数的初始值
                e_mfk - 最小流化速度下的空隙率

                返回:
                M_cyc - 当前时刻旋风分离器中的固体滞留量
                X_cyc - 旋风分离器中各档颗粒的质量份额
                dp_cyc - 旋风分离器压降
                M_ls - 返料装置颗粒滞留量
                dp_ls - 返料装置的压降
                W_re - 从返料装置返回炉膛的质量流率
                X_ls - 从返料装置返回炉膛的宽筛分质量份额
                w_rek - k档返料流率，kg/s
                G_downk - k档离开旋风分离器的颗粒流率，kg/s
                G_flyK - k档飞灰颗粒流率，kg/s
                M_cyck - 当前时刻K档颗粒在旋风分离器中的滞留量
                M_lsk - 当前时刻K档颗粒在旋风分离器中的滞留量
                dp_sp - 立管压降，Pa
                """
    r_cyc = W_re * Num_cyc / W_fa0[i]  # 循环倍率（25~30）（注意有两个旋风分离器和返料装置）
    print("r_cyc", r_cyc)
    print("返料装置固体质量份额：",X_ls)
    R_cyc[i] = r_cyc  # 循环倍率时间序列
    total_M_cyc[i] = M_cyc
    total_M_ls[i] = M_ls
    M_cyc0 = np.copy(M_cyc)  # 当前时刻能量守恒模块计算需要
    M_ls0 = np.copy(M_ls)    # 返料装置中的初始颗粒滞留量，kg
    X_cyc0 = np.copy(X_cyc)    # 将当前时刻颗粒滞留量和质量份额作为下一时刻的输入
    dp_ls0 = np.copy(dp_ls)    # 返料装置压降迭代计算的初始值
    W_re0, X_ls0 = np.copy(W_re), np.copy(X_ls)  # 返料装置总返料量迭代计算的初始，返料装置宽筛分系数的初始值
    G_fa[22, :] = W_fa0[i] * X_fa_km[22, :]  # G_fa - 给料速率，也是二维矩阵，0维度为小室，1维度为颗粒档
    W_rek[23, :] = w_rek * Num_cyc  # 将k档返料流率给到相应小室（注意有两个旋风分离器和返料装置）
    Gd[-1, :] = Wdrk.flatten()  # 确保 Wdrk 是 1D
    # print("M_cyc",M_cyc, "X_cyc",X_cyc,"M_ls",M_ls,"dp_ls",dp_ls,"W_re",W_re,"X_ls",X_ls,"w_rek",w_rek,"G_downk",G_downk,"G_flyK",G_flyK,"M_cyck",M_cyck,"M_lsk",M_lsk)
    [carbon_content, Xca_km, Gs, Gsk] = solid.solid_mass(dp, M_km, M_km0, G_hk_up, G_hk_down, G_fa,
                                                W_rek, Gd, G_outk, R_char, Xfac_km, Xc_km0, R_ca, Xfaca_km,
                                                    Xca_km0, G_downk*Num_cyc, G_flyK*Num_cyc, M_cyck*Num_cyc, M_cyck0*Num_cyc, w_rek*Num_cyc, M_lsk*Num_cyc, M_lsk0*Num_cyc)
    X_c = np.sum((1 - e_k) * X_km.reshape(1, -1) * carbon_content[2:, :], axis=1) / (1 - e_k_ksf) * 100  # 各小室含碳量（%）
    # X_cc = np.sum(X_km * carbon_content[2:, :], axis=1) * 100
    # print(f"00. 各小室含碳量百分比 （简单求和）(%): {X_cc}")
    print(f"0. 各小室含碳量百分比 (%): {X_c}")
    print(f"2. 回料阀压降 (Pa): {dp_ls}")
    print(f"2. 立管压降 (Pa): {dp_sp}")
    print(f"3. 炉膛出口夹带量: {np.sum(G_outk):.4f} kg/s")
    print(f"4. 旋风分离器返料固体颗粒: {np.sum(G_downk) * Num_cyc:.4f} kg/s")
    print(f"5. 飞灰固体颗粒: {np.sum(G_flyK) * Num_cyc:.4f} kg/s")
    print(f"6. 旋风分离器内固体滞留量: {M_cyc * Num_cyc:.4f} kg")
    print(f"7. 返料装置内固体滞留量: {M_ls * Num_cyc:.4f} kg")
    print(f"8. 各小室固体沉降量 (kg/s): {Gs}")
    print(f"8. 各小室固体向上流率 (kg/s): {W_p}")
    print(f"8. 输送失效高度以上固体流率 (kg/s): {np.sum(G_TDHk*X_km)*A_bed}")
    """
                固体质量平衡模块

                参数:
                dp - 颗粒粒径,主程序输出
                M_km - 当前时刻各小室灰质量
                M_km0 - 上一时刻各小室灰质量
                # Gsk0 - 上一时刻小室N中颗粒档k的沉降量，固体质量守恒模型自己输出
                G_hk_up - K档颗粒各小室上界面向上混合的质量流率，kg/s，0维度为小室，1维度为颗粒档,此处把密相区和稀相区组合起来,最上端控制体为离开主燃烧流率，对应流动模型输出G_hk_up，G_hk_down
                G_hk_down - K档颗粒各小室下界面向上混合的质量流率，kg/s，0维度为小室，1维度为颗粒档,此处把密相区和稀相区组合起来,对应G_hk_down
                G_fa - 给料速率，也是二维矩阵，0维度为小室，1维度为颗粒档，kg/s,边界条件，主程序输入
                W_rek - 从返料装置返料速率，kg/s，cyc模块输出到主程序 ，二维矩阵，0维度为小室，1维度为颗粒档
                Gd - 各小室排渣速率，kg/s，只有最下端控制体有排渣（流动模型输出）
                G_outk - K档颗粒离开主燃室的流率， kg/s,流动模型输出
                R_char - k档颗粒在各小室中的碳反应速率，kg/s
                Xfac_km - 给料中碳的质量百分数,已知条件
                Xc_km0 - 上一时刻碳的质量分数，维度为N+2，包括旋风分离器和返料装置 
                R_ca - k档颗粒在各小室中的氧化钙反应速率，kg/s
                Xfaca_km - 给料中氧化钙的质量百分数,已知条件
                Xca_km0 - 上一时刻氧化钙的质量分数，维度为N+2，包括旋风分离器和返料装置 
                G_downk - 旋风分离器k档颗粒当前时刻进入返料装置流率,CYC模块输出到主程序  
                G_flyK - 旋风分离器k档颗粒当前时刻飞灰流率,CYC模块输出到主程序 
                M_cyck - 旋风分离器k档颗粒当前时刻滞留量,CYC模块输出到主程序
                M_cyck0 - 旋风分离器k档颗粒上一时刻滞留量
                w_rek - 返料装置返回k档颗粒流率,CYC模块输出到主程序，一维矩阵，0维度为颗粒档 
                M_lsk - 返料装置k档颗粒当前时刻滞留量,CYC模块输出到主程序
                M_lsk0 - 返料装置k档颗粒上一时刻滞留量

                返回:
                X_km - 当前时刻灰的质量分数，维度为N，不包括旋风分离器和返料装置 
                carbon_content - 当前时刻碳的质量分数，维度为N+2，包括旋风分离器和返料装置 
                Xca_km - 当前时刻氧化钙的质量分数，维度为N+2，包括旋风分离器和返料装置  
                Gs - 每个小室的总沉降量，kg/s
                Gsk - 各小室各档灰的沉降量，kg/s
                """
    Xc_km0 = np.copy(carbon_content)
    Xca_km0 = np.copy(Xca_km)                 # 上一时刻氧化钙的质量分数，维度为N+2，包括旋风分离器和返料装置
    M_cyck0 = np.copy(M_cyck)  # 上一时刻旋风分离器k档颗粒滞留量
    M_lsk0 = np.copy(M_lsk)  # 上一时刻返料装置k档颗粒滞留量


    ## 调用传热模型，得到各小室（考虑旋风分离器和返料装置）对应的总传热系数（房德山）（考虑旋风分离器和返料装置）
    rho_susp_cyc = M_cyc / v_cyc  # 旋风分离器内固体质量浓度
    rho_susp_ls = M_ls / v_ls  # 返料装置内固体质量浓度
    rho_susp = M_p / v_cabs  # 截面平均固体颗粒质量浓度，kg/m3（小室内固体颗粒质量/小室体积）
    # 将旋风分离器和返料装置的数据插入到各列表头部（顺序：返料装置 -> 旋风分离器）
    # X_km = np.insert(X_km, 0, X_cyc, axis=0)  # 插入 X_cyc
    # X_km = np.insert(X_km, 0, X_ls, axis=0)  # 插入 X_ls
    rho_susp = np.insert(rho_susp, 0, rho_susp_cyc)
    rho_susp = np.insert(rho_susp, 0, rho_susp_ls)
    epsilon_p_cl = np.insert(epsilon_p_cl, 0, (1 - np.sum(e_mfk)))
    epsilon_p_cl = np.insert(epsilon_p_cl, 0, (1 - np.sum(e_mfk)))
    epsilon_p_c = np.insert(epsilon_p_c, 0, (1 - np.sum(e_mfk)))
    epsilon_p_c = np.insert(epsilon_p_c, 0, (1 - np.sum(e_mfk)))

    # 调用传热模型
    [heat_w, heat_t] = hr.heat_transfer_cabs(N_dil, X_km, T_secondary_w, T_cyc_w, u0, U_mfk, T_secondary_den, e_mfk,
                                             T_w, T_cor, T_ann,
                                             epsilon_p_cl, rho_susp, T_den, epsilon_p_c, T_t, mu_g, u_tk, cp_g,
                                             thermo_g, K_g, rho_g)
    """
                传热模型

                参数:
                N_dil - 稀相区最后一个小室编号
                X_km - 灰的质量分数
                T_secondary_w - 副床壁面温度，K
                T_cyc_w - 旋风分离器处壁温，K
                u0 - 表观气速，m/s
                U_mfk - 最小流化速度，m/s
                T_secondary_den - 副床温度，K
                e_mfk - 最小流化孔隙率
                T_w - 水冷壁温度 [K]（汽水系统输出）
                T_cor - 核心区温度 [K]
                T_ann - 环形区温度 [K]
                epsilon_p_cl - 颗粒团的固体体积份额（与环形区内的固体体积份额相等）
                rho_susp - 截面平均固体颗粒质量浓度，kg/m3
                T_den - 密相区温度 [K]
                epsilon_p_c - 核心区固体体积份额
                T_t - 屏式换热器温度 [K]（汽水系统输出）
                mu_g - 气体动力粘度，Pa s
                u_tk - 平均直径为dp的固体的终端速度 [m/s]
                A_w - 各小室与水冷壁的传热面积 [m²]
                A_t - 各小室与屏式换热器的传热面积 [m²]
                cp_g - 气体定压比热容，J/(mol K)
                thermo_g - 密相区气体导热系数
                K_g - 气膜（根据环形区温度和壁面温度平均值计算）的导热系数
                rho_g - 气体颗粒的密度，根据温度、压力进行修正，kg/m3

                返回:
                heat_w - 各小室与水冷壁之间的传热系数，W/(m2 K)
                heat_t - 各小室与屏过之间的传热系数，W/(m2 K)
                """
    total_heat_w[i, :] = heat_w[2:]
    print("各小室水冷壁和屏过传热系数：", heat_w, heat_t)
    # print("恒定温度下传热量(MW)：",(heat_w[2:]*A_w[2:]*(T[2:]-T_w[2:])+heat_t[2:]*A_t[2:]*(T[2:]-T_t[2:]))/10**6)

    ## 调用能量守恒方程（倪维斗）（考虑旋风分离器和返料装置）
    G_d = np.sum(Gd, axis=1)  # 排渣
    G_fly = np.zeros(N)
    G_fly[1] = np.sum(G_flyK) * 2  # 飞灰
    # M_p_0 = np.insert(M_p_0, 0, M_cyc0 * Num_cyc)
    M_p_0 = np.insert(M_p_0, 0, total_M_cyc[i-1] * Num_cyc)
    M_p_0 = np.insert(M_p_0, 0, total_M_ls[i-1] * Num_cyc)
    M_g_0 = np.insert(M_g_0, 0, 0.0)
    M_g_0 = np.insert(M_g_0, 0, 0.0)
    # M_p = np.insert(M_p, 0, M_cyc * Num_cyc)
    M_p = np.insert(M_p, 0, M_cyc * Num_cyc)
    M_p = np.insert(M_p, 0, M_ls * Num_cyc)
    M_g = np.insert(M_g, 0, 0.0)
    M_g = np.insert(M_g, 0, 0.0)
    W_p = np.insert(W_p, 0, np.sum(G_downk) * Num_cyc)
    # W_p = np.insert(W_p, 0, np.sum(G_outk)-np.sum(G_flyK)*2)
    W_p = np.insert(W_p, 0, W_re * Num_cyc)
    W_p[2] = np.sum(G_outk)
    G_d = np.insert(G_d, 0, 0)
    G_d = np.insert(G_d, 0, 0)
    W_g = np.insert(W_g, 0, W_g[0])
    W_g = np.insert(W_g, 0, 0.0)
    Gs = np.insert(Gs, 0, 0.0)
    Gs = np.insert(Gs, 0, 0.0)
    heat_cabs = np.insert(heat_cabs, 0, 0.0)
    heat_cabs = np.insert(heat_cabs, 0, 0.0)

    # 调用能量守恒模型
    [T, q_heat_w, q_heat_t] = eng.energy_conservation(M_p_0, M_g_0, T0, M_p, M_g, W_p, G_d, G_fly, W_g, W_air[i, :],
                                                      T_air[i, :], W_fa[i, :], T_fa[i, :],
                                                      W_re * Num_cyc, Gs, heat_cabs, heat_w, heat_t, A_w, A_t, T_w, T_t, dt)
    """
                能量守恒模型

                参数:
                M_p_0 - 上一时刻各小室固体质量，kg
                M_g_0 - 上一时刻各小室气体总摩尔数，mol
                T0 -  上一时刻各小室温度，K
                M_p - 各小室固体总质量，kg
                M_g - 各小室气体总摩尔数，mol
                W_p - 各小室上界面固体流率，kg/s
                G_d - 排渣，kg/s
                G_fly - 飞灰固体流率，kg/s
                W_g - 各小室气体流率，mol/s
                W_air - 各小室给风流率，mol/s
                T_air - 各小室给风温度，K
                W_fa - 各小室给煤流率，kg/s
                T_fa - 各小室给煤温度，K
                W_re - 再循环固体流量，kg/s （固体模块输出）
                Gs - 各小室总沉降量（返混量），kg/s（固体模块输出）
                heat_cabs - 各小室（燃烧）化学反应产生的热量，W（气体模块输出）
                heat_w - 各小室与水冷壁之间的传热系数，W/(m2 K)（传热模型输出）
                heat_t - 各小室与屏过之间的传热系数，W/(m2 K)（传热模型输出）
                A_w - 各小室水冷壁传热面积，m2
                A_t - 各小室屏过传热面积，m2
                T_w - 各小室水冷壁壁温，K（汽水系统输出）
                T_t - 各小室屏过壁温，K（汽水系统输出）
                dt - 时间步长，s

                返回:
                T - 各小室温度，K
                q_heat_w - （各小室）燃烧系统向水冷壁传热量，W
                q_heat_t - （各小室）燃烧系统向屏式换热器传热量，W
                """
    print("返料装置+旋风+炉膛温度（℃）：", T - T_ref, "炉膛内燃烧反应热和传热量（MW)：", heat_gas / 10 ** 6,
          np.sum(q_heat_w) / 10 ** 6 + np.sum(q_heat_t) / 10 ** 6)
    print("各小室向水冷壁传热量（MW）：", q_heat_w / 10 ** 6)
    print("各小室向屏过传热量（MW）：", q_heat_t / 10 ** 6)
    print("各小室传热量（MW）：", (q_heat_w+q_heat_t)/10**6)
    print("各小室燃烧反应热（MW）：", heat_cabs / 10 ** 6)
    total_T[i, :] = T  # 储存当前时刻各个小室温度，K
    current_state3 = T[2] - T_ref
    T0 = np.copy(T)
    T_cor, T_ann = np.copy(T), np.copy(T)
    T_secondary_den, T_den = np.copy(T), np.copy(T)  # 副床温度、密相区温度更新，K
    Q_heat_w[i, :], Q_heat_t[i, :] = q_heat_w, q_heat_t  # 储存当前时刻燃烧系统向水冷壁（汽包）和屏式换热器传热量，W
    # T = np.full(N, 900 + T_ref, dtype=np.float64)  # 调试用，除了传热模块其他模块不受能量守恒模块影响
    # T = np.linspace(900+T_ref, 850+T_ref, N)  # 从900递减到850，均匀分布


    # 模拟计算：生成当前时间和状态量（替换为实际计算）
    current_state1 = result_gas[2][0]*100
    current_state2 = r_cyc
    # current_state3 = T[2] - T_ref
    current_state4 = M_cyc*2    # 单个旋风分离器

    # 更新数据
    time_data.append(i)
    state_data1.append(current_state1)
    line1.set_data(time_data, state_data1)  # 更新曲线
    state_data2.append(current_state2)
    line2.set_data(time_data, state_data2)  # 更新曲线
    state_data3.append(current_state3)
    line3.set_data(time_data, state_data3)  # 更新曲线
    state_data4.append(current_state4)
    line4.set_data(time_data, state_data4)  # 更新曲线

    # 调整坐标轴范围
    ax1.relim()  # 重新计算数据范围
    ax1.autoscale_view()  # 自动调整坐标轴
    ax2.relim()  # 重新计算数据范围
    ax2.autoscale_view()  # 自动调整坐标轴
    ax3.relim()  # 重新计算数据范围
    ax3.autoscale_view()  # 自动调整坐标轴
    ax4.relim()  # 重新计算数据范围
    ax4.autoscale_view()  # 自动调整坐标轴

    # 刷新图形
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.1)  # 模拟计算延迟

end_time = time.time()  # 记录结束时间
print(f"执行时间：{end_time - start_time:.4f} 秒")  # 输出耗时

plt.ioff()  # 关闭交互模式
plt.show()  # 保持图形显示