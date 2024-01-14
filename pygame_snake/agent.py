import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000 # 每次训练的样本数
LR = 0.001      # 学习率

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # 随机选择动作的概率
        self.gamma = 0.9 # 折扣因子 作用是对未来的奖励进行衰减
        self.memory = deque(maxlen=MAX_MEMORY) # 记忆存储队列
        self.model = Linear_QNet(12, 256, 3) # 有11个状态，3个动作
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma) # 训练器
     
    # 获取当前状态
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x-20, head.y)
        point_r = Point(head.x+20, head.y)
        point_u = Point(head.x, head.y-20)
        point_d = Point(head.x, head.y+20)

        # 判断蛇头的左右上下是否有障碍物
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # 蛇头的左边是否有墙或者障碍物
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # 蛇头的右边
            (dir_l and game.is_collision(point_r)) or
            (dir_r and game.is_collision(point_l)) or
            (dir_d and game.is_collision(point_u)) or
            (dir_u and game.is_collision(point_d)),

            # 上边
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # 下边
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # 蛇头的移动方向
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # 食物的位置
            game.food.x < game.head.x,  # 左边
            game.food.x > game.head.x,  # 右边
            game.food.y < game.head.y,  # 上边
            game.food.y > game.head.y  # 食物在蛇头的下边
        ]

        return np.array(state, dtype=int)



    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) 
        # done 表示游戏是否结束，如果游戏结束，将记忆队列中的样本存入队列中



   # 从记忆中随机抽取样本训练
    def train_long_memory(self):
        #此处从memory中抽取样本训练
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # 随机抽取样本
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample) # 将样本分成5个数组
        self.trainer.train_step(states, actions, rewards, next_states, dones) # 训练多次


    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done) # 训练一次

    def get_action(self, state):
        # 随机选择动作 
        self.epsilon = 80 - self.n_games # 随着训练次数的增加，随机选择动作的概率越来越小
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon: #如果随机数小于epsilon，随机选择动作
            move = random.randint(0,2) # 随机选择动作 直走或者左右转
            final_move[move] = 1
        else:
            # 获取模型预测的动作
            state0 = torch.tensor(state, dtype=torch.float) # 将状态转换成张量
            prediction = self.model(state0)
            move = torch.argmax(prediction).item() # 获取预测结果中最大值，也就是最好的行动方向
            final_move[move] = 1

        return final_move

def train():
    plot_scores = [] # 记录每一局的得分
    plot_mean_scores = []   #平均分 
    total_score = 0     
    record = 0 
    agent = Agent()
    game = SnakeGameAI()
    while True:
        state_old = agent.get_state(game)   # 获取当前状态
        #移动
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        
        #获取下一个状态
        state_new = agent.get_state(game)
        
        #根据 reward 和 done 训练
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        # 存储变量
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # 一局结束
            game.reset()  # 从新开始游戏
            agent.n_games += 1   
            agent.train_long_memory()  # 从记忆中随机抽取样本训练
            
            if score > record:
                record = score # 记录最高分
                
            print('Game', agent.n_games, 'Score', score, 'Record:', record)
            
            # 画图
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()

