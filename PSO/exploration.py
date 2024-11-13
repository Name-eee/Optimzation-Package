import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

sns.set()
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 目标函数
def obj_func(x,y):
    # 定义函数 f(x, y)
    return np.sin(x) * np.cos(y) + 0.5 * np.cos(2 * x) * np.sin(2 * y)

# 全局初始化
xrange, yrange = [-5, 5], [-5, 5]
pscale = 1500
n_epoch = 5000
c1, c2 = 0.6, 0.3

# 网格初始化
x = np.linspace(xrange[0], xrange[1], 100)
y = np.linspace(yrange[0], yrange[1], 100)
Xgrid, Ygrid = np.meshgrid(x, y)
Zgrid = obj_func(Xgrid, Ygrid)

# 绘制等高线图
def contour(Xgrid, Ygrid, Zgrid, x, y):
    # 绘制二维等高线
    plt.figure(figsize=(8, 6))
    contour = plt.contour(Xgrid, Ygrid, Zgrid, levels=5, cmap="viridis")
    plt.colorbar(contour)
    plt.xlabel("x")
    plt.ylabel("y")
    # 绘制点
    plt.scatter(x, y, s=5, c='red')
    plt.show()

# 绘制三维函数图
def surface(Xface, Yface, Zface, x, y, z):
    # 绘制三维曲面
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Xface, Yface, Zface, cmap='viridis', edgecolor='none')
    # 绘制点
    ax.scatter(x, y, z, c='red', s=5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

def pso(func, xrange, yrange, n_epoch=100, pscale=1000, c1=1.0, c2=1.0):
    # 粒子初始化
    x = np.random.uniform(xrange[0], xrange[1], pscale)
    y = np.random.uniform(yrange[0], yrange[1], pscale)
    # 最值初始化
    pbest_x, pbest_y = x, y
    pbest_value = func(pbest_x, pbest_y)
    gbest_idx = np.argmin(func(pbest_x, pbest_y))
    gbest_x, gbest_y = pbest_x[gbest_idx], pbest_y[gbest_idx]
    gbest_value = func(gbest_x, gbest_y)
    
    # 迭代
    for epoch in tqdm(range(n_epoch)):
        pbest_module = np.sqrt((pbest_x-x)**2 + (pbest_y-y)**2)+0.0001
        pbest_velocity_x, pbest_velocity_y = (pbest_x-x)/pbest_module, (pbest_y-y)/pbest_module
        gbest_module = np.sqrt((gbest_x-x)**2 + (gbest_y - y)**2)+0.0001
        gbest_velocity_x, gbest_velocity_y = (gbest_x-x)/gbest_module, (gbest_y-y)/gbest_module
        v_x, v_y = c1*gbest_velocity_x + c2*pbest_velocity_x, c1*gbest_velocity_y + c2*pbest_velocity_y
        x = x + v_x
        y = y + v_y
        value_now = func(x, y)
        pbest_x, pbest_y = np.where(pbest_value < value_now, pbest_x, x), np.where(pbest_value < value_now, pbest_y, y)
        pbest_value = np.where(pbest_value < value_now, pbest_value, value_now)
        gbest_idx = np.argmin(pbest_value)
        gbest_x, gbest_y = pbest_x[gbest_idx], pbest_y[gbest_idx]
        gbest_value = func(gbest_x, gbest_y)
        # print(f"Epoch {epoch}: gbest={gbest_value}")
    return pbest_x, pbest_y, pbest_value, [gbest_x, gbest_y, gbest_value]

if __name__ == '__main__': 
    # 绘初始图
    contour(Xgrid, Ygrid, Zgrid, [], [])
    surface(Xgrid, Ygrid, Zgrid, [], [], [])
    x = np.random.uniform(xrange[0], xrange[1], pscale)
    y = np.random.uniform(yrange[0], yrange[1], pscale)
    z = obj_func(x, y)
    contour(Xgrid, Ygrid, Zgrid, x, y)
    surface(Xgrid, Ygrid, Zgrid, x, y, z)
    # PSO优化
    x_pso, y_pso, z_pso, gbest= pso(obj_func, xrange, yrange, n_epoch, pscale)
    print(f"gbest_x={gbest[0]}, gbest_y={gbest[1]}, gbest_value={gbest[2]}")
    # 绘结果图
    contour(Xgrid, Ygrid, Zgrid, x_pso, y_pso)
    surface(Xgrid, Ygrid, Zgrid, x_pso, y_pso, z_pso)
    # 直方图
    sns.histplot(z_pso, bins=30, kde=True)
    plt.title('频数直方图')
    plt.show()
    
    # gbest处理
    gbests = []
    for i in range(1000, 20001, 1000):
        _, _, _, gbest= pso(obj_func, [-5, 5], [-5, 5], n_epoch=i)
        gbests.append(gbest[2])
        
    data=gbests
    max_value = np.max(data)
    min_value = np.min(data)
    variance = np.var(data)
    print(f'最大值: {max_value}')
    print(f'最小值: {min_value}')
    print(f'方差: {variance}')

    plt.plot(list(range(1000, 20001, 1000)), data, marker='o', linestyle='-', color='b', label='数据曲线')
    plt.axhline(max_value, color='r', linestyle='--', label='最大值')
    plt.axhline(min_value, color='g', linestyle='--', label='最小值')
    plt.title(f'n_epoch - gbest曲线图')
    plt.xlabel('n_epoch')
    plt.ylabel('gbest')
    plt.ylim(-1.30, -1.29)
    plt.xticks(range(1000, 20000, 4000))
    plt.legend()
    plt.grid(color='white', linestyle='-', linewidth=0.8)  # 添加网格
    plt.gca().set_axisbelow(True)  # 确保网格在数据线下方
    # 显示图形
    plt.show()
    