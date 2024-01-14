import torch
import torch.nn as nn   
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__() 
        self.linear1 = nn.Linear(input_size, hidden_size) # 输入层
        self.linear2 = nn.Linear(hidden_size, output_size) # 输出层

    def forward(self, x): # 前向传播
        x = F.relu(self.linear1(x)) # 激活 
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'): # 保存模型
        model_folder_path = './model'
        if not os.path.exists(model_folder_path): # 如果不存在就创建
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name) # 保存模型的参数   

class QTrainer: 
    # 训练器
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma 
        self.model = model 
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr) # 优化器
        self.criterion = nn.MSELoss() # 损失函数

    def train_step(self, state, action, reward, next_state, game_over):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action =  torch.tensor(action, dtype=torch.long)
        reward =  torch.tensor(reward, dtype=torch.float)

        if(len(state.shape) == 1):
            # 如果是一维的，就转成二维的
            # unsqueeze(0) 在第0维增加一个维度
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action =  torch.unsqueeze(action, 0)
            reward =  torch.unsqueeze(reward, 0)
            game_over = (game_over, )

        # 计算预测值
        pred = self.model(state)

        # 计算目标Q值 Q_new = r + gamma * max(next_pred) 
        # Q值是预测值，Q_new是目标值
        target = pred.clone()
        for idx in range(len(game_over)):
            Q_new = reward[idx]
            if not game_over[idx]:
                # 如果游戏没有结束，就加上gamma * max(next_pred)
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx])) 
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 训练
        self.optimizer.zero_grad() # 梯度清零
        loss = self.criterion(target, pred) # 计算损失
        loss.backward() # 反向传播
        self.optimizer.step()   # 更新权重
        
