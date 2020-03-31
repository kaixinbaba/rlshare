
# 深度强化学习笔记
    传统强化学习例如TD学习，蒙特卡洛等，在处理连续动作状态空间，亦或者很大的离散动作状态空间上都会感到力不从心，因为都是基于值优化的强化学习方法，学习的方法都是建立一个表格来关联上每一个状态下所有动作的期望奖励，之后用此表格来获取最优策略。所以针对连续动作状态空间，基于神经网络的强化学习方法——深度强化学习就呼之欲出了。
## 基于值的深度强化学习
    把传统强化学习方法中的Q-learning修改为使用神经网络实现，就出现了第一个深度强化学习方法Deep-Q-Net，简称DQN，也是使用该算法DeepMind在Atari的游戏上展现了超越人类水平的技巧
### DQN
先来看看Q-learning的公式
![68f50f15798db4dd956fce9fcecd83bc.png](evernotecid://2A4F6290-ADEE-4987-9A13-0007F7180CB5/appyinxiangcom/22820264/ENResource/p14)

以下都用**CartPole-v0**（简称env）作为演示环境
**pytorch**作为代码实现框架
#### 仅仅使用单个神经网络的DQN
[仅仅使用单个神经网络 Github代码示例链接](https://github.com/kaixinbaba/hello-drl/blob/master/0_dqn_just_nn.py)
env的state拥有4个数值，action只有两个则是对应的0和1（向左和向右）
所以构建的神经网络的结构是（假设这里只用一个隐藏层16个节点）
4-16-2
所以最外层的伪代码应该是这样的：

```python
# 1 初始化环境和Agent
env = gym.make('CartPole-v0').unwrapper
agent = Agent()
# 2 循环最大训练次数
for e in range(MAX_EPISODE):
    # 初始化环境
    state = env.reset()
    # 循环 
    while True:
        # 渲染（可选）
        env.render()
        # 智能体根据当前状态选择一个动作
        action = agent.choose_action(state)
        # 拿选定的动作和环境交互，来获取下一个状态，奖励等
        next_state, reward, done, info = env.step(action)
        if done:
            # 如果结束，跳出循环，进入下一个训练迭代
            break
        # 否则将下一个状态作为当前状态，继续循环
        state = next_state

```
Agent选择动作代码片段
```python
    def choose_action(state):
        # 此时未处理的state是一个1维的数组，有4个数值 eg:[0.12, 0.43, 0.35, 0.56]
        # 所以首先需要转成torch的张量以及扩充为二维的数组
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        # 因为策略采用的是epsilon-greedy策略
        # 随机数小于epsilon
        if np.random.random() < epsilon:
            # 选择一个随机动作
        else:
            # 先通过正向传播获取当前状态下所有动作期望奖励的向量
            actions = net(state)
            # 选择期望奖励最大的动作，gym环境返回值需要是一个实数
            return torch.argmax(actions, 1).numpy()

```

---
Agent learn代码片段
```python
    def learn(s, s_, a, r):
        # 由于当前算法只使用了一个神经网络所以使用单步学习q-learning更新参数
        # 入参都不是tensor需要转换成tensor，并且由于要使用神经网络所以记得维度需要是二维的，此处非常要注意维度的问题，维度不匹配的话，有可能程序不报错，但是就无法学习到东西了
        s = change_to_tensor(s).unsqueeze(0)
        s_ = ...
        a = ...
        r = ...
        # 获取当前状态的action值
        actions = net(s)
        # 获取下一个状态的action值
        next_actions = net(s_)
        # 并获取最大动作值
        max_actions = get_max_actions(next_action)
        # 获取当前选择的动作的值
        current_action_value = actions.gather(1, a)
        # 计算目标值，使用最大动作值，本步骤就是q-learning最重要的更新步骤了
        target_action_value = r + gamma * max_actions
        
        # 计算误差
        loss = get_loss(current_action_value, target_action_value)
        # 反向传递
        opt.zero_grad()
        loss.backward()
        opt.step()  
        
```
试着学习300步，并输出可视化图
![aaf6bbe1a14b194119898dfc2f40b877.png](evernotecid://2A4F6290-ADEE-4987-9A13-0007F7180CB5/appyinxiangcom/22820264/ENResource/p2)
可以看到随着训练步数（episode）的上升，整体的奖励值在提高，
但一直到150步左右才慢慢有了上升到趋势
本算法还不是DQN最好的算法，让我们进入下一个改进算法
#### 加入 经验回放 的 DQN
先来看看 伪代码图
![f8235d9008dae33aa9494312d219cc39.png](evernotecid://2A4F6290-ADEE-4987-9A13-0007F7180CB5/appyinxiangcom/22820264/ENResource/p3)
该流程图，在之前神经网络的基础上增加了用于经验回放的**relay buffer**，将之前的交互过程和结果缓存起来，学习的时候随机从该记忆库中抽取样本进行学习，该算法用于打破之前学习上一个动作和下一个状态的相关性，并且把之前的交互过程利用起来而不是像上一个算法一样学完就丢弃。
[增加replay_buffer Github代码示例链接](https://github.com/kaixinbaba/hello-drl/blob/master/1_dqn_add_replay_buffer.py)
增加一个replay buffer，就是一个固定大小表格用来记录每次交互后的状态动作奖励
```python
class ReplayBuffer:

    def __init__(self, row_size, column_size):
        # 固定大小的数据结构就行，队列，集合，字典等都行
        # 然后再Agent初始化的时候初始化ReplayBuffer就行了
        self.memory = np.zeros([row_size, column_size])
        ...
    
    def add_memory(self, s, s_, a, r):
        # 一般是替换掉最旧的记忆
        ...
    
    def sample(self, batch_size):
        return random_sample_for_batch_size        
```
然后的改动点就是之前Agent的learn方法
```python
    def learn(self, s, s_, a, r):
        # 可以拆成两个方法，一个用来存储记忆，另一个用来学习，这里写成一个方法
        # 先存储当前的交互信息
        self.replay_buffer.add_memory(s, s_, a, r)
        if still_can_not_learn():
            # 等待记忆库的存量到达学习条件
            return
        # 学习的前提肯定是需要在记忆库中有一定量的数据
        mini_batch = self.replay_buffer.sample(s, s_, a, r)        
        s = cast_to_tensor(mini_batch)
        s_ = ...
        # 注意a指的是选择的动作的索引所以要用整型的tensor
        a = ...
        r = ...
        # 同样的在喂进神经网络前要保证变量转换成了tensor，以及确认维度是否匹配
        # 下同
        ...

```
![6c14f72a56aacd17deb54c9ca454a44a.png](evernotecid://2A4F6290-ADEE-4987-9A13-0007F7180CB5/appyinxiangcom/22820264/ENResource/p7)
这是使用原生环境的奖励值训练500次的结果，可以看到奖励峰值达到了12000，相比于之前的单神经网络进步的可不是一点点
![8787ea98886b48adc8ba5a556b717f80.png](evernotecid://2A4F6290-ADEE-4987-9A13-0007F7180CB5/appyinxiangcom/22820264/ENResource/p6)
这是使用莫烦老师demo中重新设计的奖励，训练500次的结果

#### 加入 固定目标 的 DQN
先上伪代码
![32afe2b656b7d8e2013eda856f0496e6.png](evernotecid://2A4F6290-ADEE-4987-9A13-0007F7180CB5/appyinxiangcom/22820264/ENResource/p5)
这里DQN的结构变成了两个神经网络，但是这两个神经网络结构**完全相同**，而且结合replay buffer进行SGD学习，一个eval_net每次都进行参数更新，一个target_net不主动更新参数，每隔一段时间从eval_net同步最新的参数，学习的时候下一个状态的期望奖励使用的是target_net输出的动作向量
同样先介绍下，此算法对于上面算法的改动点
[增加target_net Github代码示例链接](https://github.com/kaixinbaba/hello-drl/blob/master/2_dqn_add_target_net.py)
首先Agent在初始化的时候要额外初始化一个神经网络
```python
    def __init__(self):
        ...
        # 初始化两个相同的神经网络
        self.eval_net = net()
        self.target_net = net()
        ...
```
然后就是学习的时候的区别了
```python
    def learn(self, s, s_, a, r):
        # 新增一个超参数，控制什么时候 targe_net 需要 同步 eval_net的参数
        # 就是论文伪代码中的C
        if every_C_step:
            # 从 eval_net 同步到 target_net 的参数
            sync_the_targe_net_param()
        ...
        # 当前状态仍然从eval_net获取期望奖励
        actions = eval_net(s)
        # 但下一个状态从target_net获取期望奖励,并且要冻结target_net的参数，不让它自动学习
        next_actions = target_net(s_).detach()
        #下同
        ...

```
![22f231495dc02a3710b71c3709a9acd2.png](evernotecid://2A4F6290-ADEE-4987-9A13-0007F7180CB5/appyinxiangcom/22820264/ENResource/p9)
这是训练500次后的奖励图

#### Double DQN
传统dqn**有过估**计的问题，原因是在评估的策略用的是**贪婪策略max**，所以double dqn（以下简称DDQN）在评估的时候做了稍稍的改动，极大的解决了过估计的问题
先来看看伪代码
![0e166ef8c5d1a8f4d3c5081a27861663.png](evernotecid://2A4F6290-ADEE-4987-9A13-0007F7180CB5/appyinxiangcom/22820264/ENResource/p11)
DDQN相对于之前的DQN最大的改动我用红框标注出来了，就是评估下一个状态的最大动作用的是eval_net，而动作值选的是target_net的动作值
[DDQN Github代码示例链接](https://github.com/kaixinbaba/hello-drl/blob/master/3_double_dqn.py)
代码改动的话就learn的时候的改动
```python
    def learn(...):
        ...
        # 当前状态仍然从eval_net获取期望奖励
        actions = eval_net(s)
        # 这步最重要，就是获取eval_net里下一个状态s_对应的最大动作值的索引
        next_max_action_from_eval_index = eval_net(s_).max(1)[1].unsqueeze(1)
        # 同dqn
        next_actions_from_target = target_net(s_).detach()
        # 但是这里取具体的动作值用的是从eval_net中获取的索引
        max_actions = next_actions_from_target.gather(1, next_max_action_from_eval_index)
        # 更新步骤同DQN
        target_action_value = r + gamma * max_actions
        # 下同
        ...
```
![00902c9a4ae9109a114a12f75743ec9c.png](evernotecid://2A4F6290-ADEE-4987-9A13-0007F7180CB5/appyinxiangcom/22820264/ENResource/p12)
这是使用同样参数跑的500次训练结果，可以看到模型收敛的速度变快了，提前到了300多步时，就能获去1700多的奖励，优于DQN的400多步

#### Double DQN using priority replay buffer
这次改进的是replay buffer了，不再是完全随机的去选择之前的交互的memory了，而是对这些transaction标记上了优先级，而在sample的时候根据优先级增加的权重去选择mini-batch，我用自己的方式实现了简单版的和论文中的sumtree不一样，但是更好理解一点
##### 01 只用了td-error的绝对值作为优先级的选择标准
[DDQN_use_priority Github代码示例链接](https://github.com/kaixinbaba/hello-drl/blob/master/4_prioritize_dqn_01.py)
```python
class Transaction:
    def __init__(self, s, a, r, s_, tderror, index):
        # 前4个参数就不解释了
        # tderror是这次增加的新参数需要一起存进记忆库
        # 公式就是 abs(target_q - eval_q)
        # 但是target_q 也是通过eval_net得到的
        # index 则是标记当前这条transaction是出于记忆库中的位置，
        # 方便之后更新tderror
        self.s = s
        ...

class PrioritizeReplayBuffer:
    # 代替之前的ReplayBuffer
    def add_memory(self, s, s_, a, r, tderror):
        if self.memory_count < self.row_size:
            # 直接创建transaction，并添加
            self.memory.append(new_transaction)
        else:
            # 获取最后一个transaction的tderror
            # 因为会排序，所以最后一个的tderror是最小的
            old_tderror = get_last_tderror()
            if tderror > old_tderror:
                # 如果当前的tderror比较多，则替换最后一个
                self.memory[-1] = transaction
        self.memory_count += 1
        # 重要！需要排序来保证最后一个transaction是最小tderror
        self.sort_memory()
     def sort_memory(self):
        # 对整个memory根据优先级进行排序
        
     def sample(self, batch_size=32):
        # 根据权重概率，选择batch_size个sample去学习

    def get_sample_prop(self):
        # 根据存储的sample的优先级，获取权重概率
        # 该返回的权重概率，和必须为1，并且长度和当前的memory_size是一样的
        
class Agent:

    def learn(self, s, a, r, s_):
        # 获取当前的tderror，并存到记忆库中
        tderror = get_tderror(s ,a, r, s_)
        self.replay_buffer.add_memory(...)
        ...
        mini_batch = self.replay_buffer.sample()
        # 解包的过程和之前的还是不一样的
        ...
        # 重要！在反向传播后，需要再次获取mini_batch中的tderror，并更新回记忆库
        update_new_tderror()
       
```
![9167732a3dbf8aeb4345d55d4963da7e.png](evernotecid://2A4F6290-ADEE-4987-9A13-0007F7180CB5/appyinxiangcom/22820264/ENResource/p15)
是用同样的参数和模型跑完的效果，感觉效果略好一点点
##### 02 在01的基础上增加了两个新的超参数a和b
加了这两个超参数，相当于在优先选择和随机选择上平衡了下，以及降低优先级高的子集对整体学习的影响，减少了对子集的过拟合
![9ed880b1d84c7fa254b791d0785fdfa5.png](evernotecid://2A4F6290-ADEE-4987-9A13-0007F7180CB5/appyinxiangcom/22820264/ENResource/p16)

```python
#增加了三个超参数
a = 0.6
b = 0.4
b_increment = 0.001

class PrioritizeReplayBuffer:
    def get_sample_prop()
        # 上图左边体现
        sum_tderror = np.sum([np.power(t.tderror, a) for t in self.memory])
        result = [np.power(t.tderror, a) / sum_tderror for t in self.memory]
        return result
        
        
class Agent:
    
    def learn(self):
        ...
        # 上图右边体现
        # 这里就不能用之前的Loss MSE了要自己手算一个loss公式
        b = update_with_increment_max_1
        new_loss = torch.mean(torch.FloatTensor(np.power(memory_size*prop, -b))*square_diff_loss_without_mean)
        # 拿这个new_loss去反向传播
        ...     
```
![1c562fb03743b3160f53a3d189ecc417.png](evernotecid://2A4F6290-ADEE-4987-9A13-0007F7180CB5/appyinxiangcom/22820264/ENResource/p17)
同样的参数训练500次的结果
#### Dueling DQN
##### 01 Q=V+A
是把原先神经网络输出的Q值拆分成了V和A，而新的Q则是取两者的和，像这样：
![de85e6cc490956aa63ca65ed910eca0b.png](evernotecid://2A4F6290-ADEE-4987-9A13-0007F7180CB5/appyinxiangcom/22820264/ENResource/p18)
示意图是：
![8b673d00633bc34f0d3f27bc52fdb09d.png](evernotecid://2A4F6290-ADEE-4987-9A13-0007F7180CB5/appyinxiangcom/22820264/ENResource/p19)
[Dueling_DQN Github代码示例链接](https://github.com/kaixinbaba/hello-drl/blob/master/5-dueling_dqn_01.py)
代码的部分的话只要修改Net初始化和正向传播即可

```python
class Net(nn.Module):

    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.h1 = nn.Linear(input_size, 16)
        self.h1.weight.data.normal_(0, 0.1)
        self.A = nn.Linear(16, output_size)
        self.A.weight.data.normal_(0, 0.1)
        self.V = nn.Linear(16, 1)
        self.V.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.h1(x))
        A = self.A(x)
        V = self.V(x)
        out = V + A
        return out

```
![ac0a9f269b2fd2e13f789f1754d81c06.png](evernotecid://2A4F6290-ADEE-4987-9A13-0007F7180CB5/appyinxiangcom/22820264/ENResource/p20)
这里修改成了跑200个episode，因为很快就收敛了
##### 02 Q=V+(A-mean(A))
稍微变化下就是把A标准化一点，让它训练的过程更加稳定平滑
公式如下：
![07e5b9583d8f30008bcac06c80124a8a.png](evernotecid://2A4F6290-ADEE-4987-9A13-0007F7180CB5/appyinxiangcom/22820264/ENResource/p21)
```python
    def forward(self, x):
        ...
        out = V + (A - torch.mean(A))
        ...
```
![77e638b75f6f3b7a2b0cac834a11f63d.png](evernotecid://2A4F6290-ADEE-4987-9A13-0007F7180CB5/appyinxiangcom/22820264/ENResource/p22)

## 基于策略的深度强化学习

### Policy Gradient
policy gradient直接输出了动作的选择概率，
所以在神经网络最后一层输出需要用softmax作为激活函数
先看看伪代码
![fef939db637b9633b9236a26b4825d9e.png](evernotecid://2A4F6290-ADEE-4987-9A13-0007F7180CB5/appyinxiangcom/22820264/ENResource/p23)

```python
    def forward(self, x):
        x = F.relu(self.hidden(x))
        out = self.out(x)
        out = F.softmax(out, dim=-1)
        return out
```
而且原始版的policy gradient是on-policy，学习的序列就是交互的序列，也没有类似于replaybuffer的结构来存储过往的经验，并且都是在回合结束才能学习
```python
    ...
    s_, r, done, _ = env.step(a)   
    # 这里store和replaybuffer是完全不一样的
    # 这里的经验只存当前episode，并且用完就丢弃了
    agent.store(s, a, r)
    if done:
        # 回合结束才进行学习
        agent.learn()
```
store方法就是用3个列表把当前的数据存下来而已
```python

    def __init__(self, n_s, n_a):
        ...
        self.states = []
        self.actions = []
        self.rewards = []
        
    def store(self, s, a, r):
        self.states.append(torch.FloatTensor(s))
        self.actions.append(a)
        self.rewards.append(r)
```
learn就是最重要的学习方法了
```python
    def learn(self):
        input_states = torch.FloatTensor(np.vstack(self.states))
        action_indexes = torch.LongTensor(np.vstack(self.actions))
        # 因为self.rewards里存的是每次的即时奖励，但是参与计算时需要乘上gamma
        rewards = self._discount_rewards()
        # 获取当前所有状态的动作概率
        actions_prop = self.net(input_states)
        # 然后选择本次episode真正采取的动作对应的概率
        actions = torch.gather(actions_prop, 1, action_indexes)
        # 负号因为 框架只能用梯度下降，但是我们这里要求梯度上升，所以取反
        # 这里的log * r，就是 公式里的log * vt
        loss = torch.mean(-torch.log(actions) * rewards)

        self.optim_func.zero_grad()
        # back
        loss.backward()
        self.optim_func.step()
        # 每次学完 清空列表
        self.states = []
        self.actions = []
        self.rewards = []
        return sum(rewards)
```
![c2404cfc650bb88a6fe1f39310648c2c.png](evernotecid://2A4F6290-ADEE-4987-9A13-0007F7180CB5/appyinxiangcom/22820264/ENResource/p24)
这里是训练300次的奖励曲线（ps：使用的是原始奖励）
### Actor-Critic

