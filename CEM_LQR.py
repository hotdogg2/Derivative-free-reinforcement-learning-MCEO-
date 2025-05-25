import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import control


# 定义LQR模型类
class LQR_Model:
    def __init__(self, dt=1):
        self.dt = dt
        A, B, Q, R = self.system()
        C = np.eye(A.shape[0])
        D = np.zeros((A.shape[0], B.shape[1]))

        sys = control.ss(A, B, C, D)
        sysd = sys.sample(dt)
        A, B, C, D = control.ssdata(sysd)
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

    def predict(self, x, u):
        return np.asarray(np.matmul(self.A, x) + np.matmul(self.B, u))

    def compute_cost(self, x, u):
        return np.sum(x*np.matmul(self.Q, x), axis=0) + np.sum(u*np.matmul(self.R, u), axis=0)

    def system(self):
        A = np.array([[1, 1],
                      [0, 1]])
        B = np.array([[1],
                      [1]])
        Q = np.array([[1, 0],
                      [0, 0]])
        R = np.array([0.3])

        return A, B, Q, R

    def dsystem(self):
        return self.A, self.B, self.Q, self.R



# 定义神经网络结构
class Net(nn.Module):
    def __init__(self, units):
        super(Net, self).__init__()
        self.units = units
        self.fc1 = nn.Linear(N_STATES, self.units)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc1.bias.data.normal_(0, 0.1)
        self.out = nn.Linear(self.units, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)
        self.out.bias.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        actions_value = self.out(x)
        return actions_value


def lqr_evaluation(policy, lqr_model, x0, N):
    total_cost = 0
    x = x0
    for t in range(N):
        u = policy(torch.tensor(x, dtype=torch.float32)).detach().numpy()
        x_next = lqr_model.predict(x, u)
        total_cost += lqr_model.compute_cost(x_next, u)
        x = x_next
    return total_cost

def mceo(net, objective_function, num_averages=5, agents_per_average=10, max_iterations=50, initial_std=1.0):
    param_size = sum(p.numel() for p in net.parameters())
    search_space = np.array([[-1, 1]] * param_size)  # 假设参数范围 [-5, 5]

    # 初始化 Averages 均匀分布在搜索空间
    averages = np.random.uniform(search_space[:, 0], search_space[:, 1], (num_averages, param_size))
    sigma_global = 2 * (search_space[:, 1] - search_space[:, 0]) / num_averages

    best_score = np.inf
    best_policy = None
    best_scores = []
    s = []

    for iteration in range(max_iterations):
        search_agents = []
        scores = []

        for i in range(num_averages):
            sigma_i = ((i + 1) / num_averages) ** 3 * sigma_global
            agents = np.random.randn(agents_per_average, param_size) * sigma_i + averages[i]
            search_agents.extend(agents)

        # 裁剪到合法范围
        search_agents = np.clip(search_agents, search_space[:, 0], search_space[:, 1])

        for agent in search_agents:
            # 注入参数到 policy 网络
            policy = Net(net.units)
            idx = 0
            for param in policy.parameters():
                shape = param.data.shape
                num = param.numel()
                param.data = torch.tensor(agent[idx:idx+num].reshape(shape), dtype=torch.float32)
                idx += num

            score = objective_function(policy)
            scores.append(score)

            if score < best_score:
                best_score = score
                best_policy = policy

        best_scores.append(best_score)

        # 综合排序
        search_agents = np.array(search_agents)
        scores = np.array(scores)
        sort_idx = np.argsort(scores)
        top_agents = search_agents[sort_idx[:num_averages]]

        # 更新 Averages
        averages = top_agents.copy()

        # SRF: 动态调整 std（靠前的减小，靠后的增大）
        SRF = np.linspace(0.95, 1.05, num=num_averages)
        sigma_global = sigma_global * SRF.mean()  # 平均缩放一次

        print(f"[MCEO] Iteration {iteration}: Best Score = {best_score:.6f}")

    plt.plot(best_scores)
    plt.xlabel("Iteration")
    plt.ylabel("Best Score")
    plt.title("MCEO Optimization Progress")
    plt.grid(True)
    plt.show()

    return best_policy, best_scores


def cem(net, objective_function, num_samples=50, num_elite=5, max_iterations=50, initial_mean=0, initial_std=1):
    mean = np.concatenate([net.fc1.weight.data.numpy().flatten(),
                           net.fc1.bias.data.numpy().flatten(),
                           net.out.weight.data.numpy().flatten(),
                           net.out.bias.data.numpy().flatten()])

    std = np.full(mean.shape, initial_std)
    best_policy = None
    best_score = np.inf
    best_plot = []
    s = []

    for iteration in range(max_iterations):
        samples = np.random.randn(num_samples, mean.size) * std + mean
        scores = []

        for i in range(num_samples):
            policy = Net(net.units)
            weight_len_fc1 = net.fc1.weight.data.numel()
            bias_len_fc1 = net.fc1.bias.data.numel()
            weight_len_out = net.out.weight.data.numel()
            bias_len_out = net.out.bias.data.numel()

            policy.fc1.weight.data = torch.tensor(samples[i, :weight_len_fc1].reshape(net.fc1.weight.data.shape),
                                                  dtype=torch.float32)
            policy.fc1.bias.data = torch.tensor(samples[i, weight_len_fc1:weight_len_fc1 + bias_len_fc1],
                                                dtype=torch.float32)
            policy.out.weight.data = torch.tensor(
                samples[i, weight_len_fc1 + bias_len_fc1:weight_len_fc1 + bias_len_fc1 + weight_len_out].reshape(
                    net.out.weight.data.shape), dtype=torch.float32)
            policy.out.bias.data = torch.tensor(samples[i, weight_len_fc1 + bias_len_fc1 + weight_len_out:],
                                                dtype=torch.float32)

            score = objective_function(policy)
            scores.append(score)

            if score < best_score:
                best_plot.append(score)
                best_score = score
                best_policy = policy

        elite_indices = np.argsort(scores)[:num_elite]
        elite_samples = samples[elite_indices]

        mean = elite_samples.mean(axis=0)
        std = elite_samples.std(axis=0)

        print(f"Iteration {iteration}: Best score = {best_score}")
        s.append(best_score)

    plt.plot(best_plot)

    return best_policy,best_plot,s






def control_system(policy, lqr_model, x0, N):
    x = x0
    states = [x0]
    controls = []
    x1_plot = []
    x2_plot = []

    for t in range(N):
        u = policy(torch.tensor(x, dtype=torch.float32)).detach().numpy()

        x1_plot.append(x[0])
        x2_plot.append(x[1])

        x_next = lqr_model.predict(x, u)
        states.append(x_next)
        controls.append(u)
        x = x_next

    return states, controls,x1_plot,x2_plot


# 参数
N_STATES = 2
N_ACTIONS = 1
UNITS = 50

lqr_model = LQR_Model(dt=1)

net = Net(units=UNITS)

best_policy1,best_plot,s1 = cem(net, lambda policy: lqr_evaluation(policy, lqr_model, x0=np.array([1, -1]), N=30), num_samples=50,
                  num_elite=12, max_iterations=50)
best_policy2,s2 = mceo(
    net,
    lambda policy: lqr_evaluation(policy, lqr_model, x0=np.array([1, -1]), N=30),

)

x0_new = np.array([1.0, -1.0])
states, controls, x1_plot1,x2_plot1  = control_system(best_policy1, lqr_model, x0_new, 30)
states, controls, x1_plot2,x2_plot2  = control_system(best_policy2, lqr_model, x0_new, 30)

# 输出控制输入 u 的变化曲线
# controls = np.array(controls).flatten()
# plt.plot(range(20), controls)
# plt.xlabel('Time Step')
# plt.ylabel('Control Input u')
# plt.title('Control Input u over Time Steps')
# plt.show()

plt.figure(figsize=(10, 5))
plt.plot(np.array(s1),label="CEM" ,color='red', linewidth=1)
plt.plot(np.array(s2),label="MCEO", color='blue', linewidth=1)

plt.yscale('log')  # <<< 对数坐标
plt.title("N_units = 60")
plt.xlabel("iterations")
plt.ylabel("Value (log scale)")
plt.grid(True, which='both')  # 显示主/次网格

plt.legend()
plt.show()



fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# 绘制x1
axs[0,0].plot( x1_plot1,color = 'red')
axs[0,0].set_xlabel('Time_step')
axs[0,0].set_ylabel('x1')
axs[0,0].set_title('CEM_x1')

# 绘制x2
axs[1,0].plot(x2_plot1,color = 'red')
axs[1,0].set_xlabel('Time_step')
axs[1,0].set_ylabel('x2')
axs[1,0].set_title('CEM_x2')

axs[0,1].plot( x1_plot2,color = 'blue')
axs[0,1].set_xlabel('Time_step')
axs[0,1].set_ylabel('x1')
axs[0,1].set_title('MCEO_X1')

# 绘制x2
axs[1,1].plot(x2_plot2,color = 'blue')
axs[1,1].set_xlabel('Time_step')
axs[1,1].set_ylabel('x2')
axs[1,1].set_title('MCEO_X2')

'''
# 绘制u
controls = np.array(controls).flatten()
axs[2].plot(controls)
axs[2].set_xlabel('Time Step')
axs[2].set_ylabel('Control Input u')
axs[2].set_title('Control Input u over Time Steps')
'''
plt.tight_layout()
plt.show()

print(best_plot)

print(x1_plot)
print(x2_plot)
print(controls)


'''
#claude
class Net(nn.Module):
    def __init__(self, units1, units2):
        super(Net, self).__init__()
        self.units1 = units1
        self.units2 = units2
        
        # 第一隐藏层
        self.fc1 = nn.Linear(N_STATES, self.units1)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc1.bias.data.normal_(0, 0.1)
        
        # 第二隐藏层
        self.fc2 = nn.Linear(self.units1, self.units2)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc2.bias.data.normal_(0, 0.1)
        
        # 输出层
        self.out = nn.Linear(self.units2, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)
        self.out.bias.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        actions_value = self.out(x)
        return actions_value


# 修改 CEM 算法中的参数串联和解串联
def cem(net, objective_function, num_samples=50, num_elite=5, max_iterations=50, initial_std=1):
    # 将所有网络参数串联成一个向量
    mean = np.concatenate([net.fc1.weight.data.numpy().flatten(),
                           net.fc1.bias.data.numpy().flatten(),
                           net.fc2.weight.data.numpy().flatten(),  # 新增：第二层权重
                           net.fc2.bias.data.numpy().flatten(),    # 新增：第二层偏置
                           net.out.weight.data.numpy().flatten(),
                           net.out.bias.data.numpy().flatten()])

    std = np.full(mean.shape, initial_std)
    best_policy = None
    best_score = np.inf
    best_plot = []
    s = []

    for iteration in range(max_iterations):
        samples = np.random.randn(num_samples, mean.size) * std + mean
        scores = []

        for i in range(num_samples):
            policy = Net(net.units1, net.units2)  # 更新为两层隐藏层
            
            # 计算各参数的长度
            weight_len_fc1 = net.fc1.weight.data.numel()
            bias_len_fc1 = net.fc1.bias.data.numel()
            weight_len_fc2 = net.fc2.weight.data.numel()  # 新增
            bias_len_fc2 = net.fc2.bias.data.numel()      # 新增
            weight_len_out = net.out.weight.data.numel()
            bias_len_out = net.out.bias.data.numel()
            
            # 索引起始位置
            idx = 0
            
            # 第一层参数
            policy.fc1.weight.data = torch.tensor(samples[i, idx:idx+weight_len_fc1].reshape(net.fc1.weight.data.shape),
                                                  dtype=torch.float32)
            idx += weight_len_fc1
            policy.fc1.bias.data = torch.tensor(samples[i, idx:idx+bias_len_fc1], dtype=torch.float32)
            idx += bias_len_fc1
            
            # 第二层参数
            policy.fc2.weight.data = torch.tensor(samples[i, idx:idx+weight_len_fc2].reshape(net.fc2.weight.data.shape),
                                                  dtype=torch.float32)
            idx += weight_len_fc2
            policy.fc2.bias.data = torch.tensor(samples[i, idx:idx+bias_len_fc2], dtype=torch.float32)
            idx += bias_len_fc2
            
            # 输出层参数
            policy.out.weight.data = torch.tensor(samples[i, idx:idx+weight_len_out].reshape(net.out.weight.data.shape), 
                                                  dtype=torch.float32)
            idx += weight_len_out
            policy.out.bias.data = torch.tensor(samples[i, idx:], dtype=torch.float32)

            score = objective_function(policy)
            scores.append(score)

            if score < best_score:
                best_plot.append(score)
                best_score = score
                best_policy = policy

        elite_indices = np.argsort(scores)[:num_elite]
        elite_samples = samples[elite_indices]

        mean = elite_samples.mean(axis=0)
        std = elite_samples.std(axis=0)

        print(f"Iteration {iteration}: Best score = {best_score}")
        s.append(best_score)

    plt.plot(best_plot)

    return best_policy, best_plot, s


# 修改 MCEO 算法中的参数解串联
def mceo(net, objective_function, num_averages=5, agents_per_average=10, max_iterations=50, initial_std=1.0):
    # 计算参数总数
    param_size = sum(p.numel() for p in net.parameters())
    search_space = np.array([[-1, 1]] * param_size)  # 假设参数范围 [-1, 1]

    # 初始化 Averages 均匀分布在搜索空间
    averages = np.random.uniform(search_space[:, 0], search_space[:, 1], (num_averages, param_size))
    sigma_global = 2 * (search_space[:, 1] - search_space[:, 0]) / num_averages

    best_score = np.inf
    best_policy = None
    best_scores = []

    for iteration in range(max_iterations):
        search_agents = []
        scores = []

        for i in range(num_averages):
            sigma_i = ((i + 1) / num_averages) ** 3 * sigma_global
            agents = np.random.randn(agents_per_average, param_size) * sigma_i + averages[i]
            search_agents.extend(agents)

        # 裁剪到合法范围
        search_agents = np.clip(search_agents, search_space[:, 0], search_space[:, 1])

        for agent in search_agents:
            # 注入参数到 policy 网络
            policy = Net(net.units1, net.units2)  # 更新为两层隐藏层
            idx = 0
            for param in policy.parameters():
                shape = param.data.shape
                num = param.numel()
                param.data = torch.tensor(agent[idx:idx+num].reshape(shape), dtype=torch.float32)
                idx += num

            score = objective_function(policy)
            scores.append(score)

            if score < best_score:
                best_score = score
                best_policy = policy

        best_scores.append(best_score)

        # 综合排序
        search_agents = np.array(search_agents)
        scores = np.array(scores)
        sort_idx = np.argsort(scores)
        top_agents = search_agents[sort_idx[:num_averages]]

        # 更新 Averages
        averages = top_agents.copy()

        # SRF: 动态调整 std（靠前的减小，靠后的增大）
        SRF = np.linspace(0.95, 1.05, num=num_averages)
        sigma_global = sigma_global * SRF.mean()  # 平均缩放一次

        print(f"[MCEO] Iteration {iteration}: Best Score = {best_score:.6f}")

    plt.plot(best_scores)
    plt.xlabel("Iteration")
    plt.ylabel("Best Score")
    plt.title("MCEO Optimization Progress")
    plt.grid(True)
    plt.show()

    return best_policy, best_scores


# 使用新的网络架构
N_STATES = 2
N_ACTIONS = 1
UNITS1 = 10  # 第一隐藏层神经元数量
UNITS2 = 8   # 第二隐藏层神经元数量

lqr_model = LQR_Model(dt=1)

# 创建双隐藏层网络
net = Net(units1=UNITS1, units2=UNITS2)

# 使用CEM优化
best_policy_cem, best_plot_cem, s1 = cem(
    net, 
    lambda policy: lqr_evaluation(policy, lqr_model, x0=np.array([1, -1]), N=20), 
    num_samples=50,
    num_elite=12, 
    max_iterations=50
)

# 使用MCEO优化
best_policy_mceo, s2 = mceo(
    net,
    lambda policy: lqr_evaluation(policy, lqr_model, x0=np.array([1, -1]), N=20),
)

# 测试最佳策略
x0_new = np.array([1.0, -1.0])
states, controls, x1_plot, x2_plot = control_system(best_policy_cem, lqr_model, x0_new, 20)

# 绘制比较图
plt.figure(figsize=(10, 5))
plt.plot(np.array(s1), color='red', linewidth=1, label='CEM')
plt.plot(np.array(s2), color='blue', linewidth=1, label='MCEO')
plt.legend()
plt.yscale('log')  # 对数坐标
plt.title("Log Scale Comparison of CEM vs MCEO")
plt.xlabel("Iteration")
plt.ylabel("Cost (log scale)")
plt.grid(True, which='both')
plt.show()
'''