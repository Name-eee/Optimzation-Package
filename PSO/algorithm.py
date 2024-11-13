import numpy as np

# 目标函数
def obj_func(x,y):
    # 定义函数 f(x, y)
    return np.sin(x) * np.cos(y) + 0.5 * np.cos(2 * x) * np.sin(2 * y)

# 全局初始化
xrange, yrange = [-5, 5], [-5, 5]
pscale = 1500
n_epoch = 5000
c1, c2 = 0.6, 0.3

def pso(func, xrange, yrange, n_epoch=100, pscale=1000, c1=1.0, c2=1.0):
    # 粒子初始化
    x = np.random.uniform(xrange[0], xrange[1], pscale)
    y = np.random.uniform(yrange[0], yrange[1], pscale)
    v_x = np.zeros_like(x)
    v_y = np.zeros_like(y)
    gbest_log = []
    # 各粒子最值初始化
    pbest_x, pbest_y = x, y
    pbest_value = func(pbest_x, pbest_y)
    # 全局最值初始化
    gbest_idx = np.argmin(func(pbest_x, pbest_y))
    gbest_x, gbest_y = pbest_x[gbest_idx], pbest_y[gbest_idx]
    gbest_value = func(gbest_x, gbest_y)
    # 迭代
    for epoch in range(n_epoch):
        # 计算向着pbest, gbest的速度
        pbest_velocity_x, pbest_velocity_y = pbest_x-x, pbest_y-y
        gbest_velocity_x, gbest_velocity_y = gbest_x-x, gbest_y-y
        # 更新速度
        v_x = v_x + c1*np.random.random()*gbest_velocity_x + c2*np.random.random()*pbest_velocity_x
        v_y = v_y + c1*np.random.random()*gbest_velocity_y + c2*np.random.random()*pbest_velocity_y
        # 更新位置
        x_tmp = x + v_x
        y_tmp = y + v_y
        x = np.where(x_tmp > xrange[0], x_tmp, xrange[0])
        y = np.where(y_tmp > yrange[0], y_tmp, yrange[0])
        x = np.where(x < xrange[1], x, xrange[1])
        y = np.where(y < yrange[1], y, yrange[1])
        # 计算适应度函数
        z = func(x, y)
        # 更新pbest
        pbest_x, pbest_y = np.where(pbest_value < z, pbest_x, x), np.where(pbest_value < z, pbest_y, y)
        pbest_value = np.where(pbest_value < z, pbest_value, z)
        # 更新gbest
        gbest_idx = np.argmin(pbest_value)
        gbest_x, gbest_y = pbest_x[gbest_idx], pbest_y[gbest_idx]
        gbest_value = pbest_value[gbest_idx]
        gbest_log.append(gbest_value)
    return pbest_x, pbest_y, pbest_value, gbest_log, [gbest_x, gbest_y, gbest_value]

if __name__ == '__main__': 
    # PSO优化
    x_pso, y_pso, z_pso, gbest_log, gbest= pso(obj_func, xrange, yrange, n_epoch, pscale)
    print(f"gbest_x={gbest[0]}, gbest_y={gbest[1]}, gbest_value={gbest[2]}")
    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(gbest_log)), )