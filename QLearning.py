import numpy as np
import pandas as pd
import time

N_STATES = 6  # 1维世界的宽度
ACTIONS = ['left', 'right']  # 探索者的可用动作
EPSILON = 0.9  # 贪婪度 greedy
ALPHA = 0.1  # 学习率
GAMMA = 0.9  # 奖励递减值
MAX_EPISODES = 13  # 最大回合数
FRESH_TIME = 0.3  # 移动间隔时间


class Env:
    def __init__(self):
        self.state = 0
        self.n_states = N_STATES
        self.step_counter = 0
        self.actions = ACTIONS

    def render(self):
        # 图像显示
        env_list = ['-'] * (N_STATES - 1) + ['T']
        if self.state == 'end':
            env_list[N_STATES - 1] = '$'
            print("\r", ''.join(env_list), "   steps   ", self.step_counter, end="")
            time.sleep(1)
        else:
            env_list[self.state] = 'o'
            interaction = ''.join(env_list)
            print("\r", interaction, end="")
            time.sleep(FRESH_TIME)

    def step(self, action):
        is_done = False
        if action == 'right':
            if self.state == N_STATES - 2:
                r = 1
                s_ = 'end'
                is_done = True
            else:
                r = 0
                s_ = self.state + 1
        else:
            if self.state == 0:
                r = 0
                s_ = self.state
            else:
                r = 0
                s_ = self.state - 1
        self.state = s_
        self.step_counter += 1
        return s_, r, is_done

    def reset(self):
        self.state = 0
        self.step_counter = 0
        return self.state


class QL:
    def __init__(self):
        self.env = Env()
        self.q_table = pd.DataFrame(
            np.zeros((self.env.n_states - 1, len(self.env.actions))),
            columns=self.env.actions,
        )

    def choose_action(self, state):
        state_action = self.q_table.iloc[state, :]
        if (np.random.uniform() > EPSILON) or (state_action.all() == 0):
            action_name = np.random.choice(self.env.actions)
        else:
            action_name = self.env.actions[state_action.argmax()]
        return action_name

    def learn(self):
        for episode in range(MAX_EPISODES):
            s = self.env.reset()
            self.env.render()
            while 1:
                a = self.choose_action(s)
                s_, r, is_end = self.env.step(a)

                # TDLearning更新表格
                q_target = r if is_end else r + GAMMA * self.q_table.iloc[s_, :].max()
                q_predict = self.q_table.loc[s, a]
                self.q_table.loc[s, a] += ALPHA * (q_target - q_predict)

                #刷新页面
                self.env.render()
                s = s_

                if is_end:
                    print('\n')
                    break


if __name__ == '__main__':
    ql = QL()
    ql.learn()
    print(ql.q_table)
