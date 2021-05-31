import gym
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import *
from rl.memory import SequentialMemory
import numpy as np
from gym import spaces
from pprint import pprint as pp


class Market(gym.Env):
    """
    仮想通貨 相場

    Attributes
    ----------
    observation:
      仮想通貨 環境
        open
        high
        low
        close
        volume
        jpy 保有日本円
        coin 保有仮想通貨数
    action :
      売り買い（金額は一旦固定）
        buy
        sell
        stay

    -
    """
    n_actions = 3
    MAX = 99,999,999 # OHLCVそれぞれの最大値
    metadata = {'render.modes': ['human']}
    INIT_JPY = 100,000
    INIT_COIN = 100

    def __init__(self):
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(low=0, high=self.MAX, shape=(7,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.reward = 0
        self.action = 0

        return np.array([0, self.goal])

    def step(self, action):
        self.cards_sum += action

        done = False
        reward = 0

        if self.cards_sum == self.goal:
            # ゴール報酬は最大値の2倍（適当）
            reward = self.MAX*2
            done = True
        elif self.cards_sum > self.goal:
            # ゴール超えたら失格
            reward = -10
        elif np.any(self.cards_list==action):
            # すでに出したカードを出したら失格
            reward = -50
            done = True
        else:
            # 目標数(goal)から近いほうが報酬が高い
            reward = action + (self.cards_sum/self.MAX) * 2

        if action != 0:
            self.cards_list = np.append(self.cards_list, action)
        self.action = action
        self.reward = reward

        info = {}

        return np.array([self.cards_sum, self.goal]), reward, done, info

    def render(self, mode='human', close=False):
        if mode != 'human':
          raise NotImplementedError()
        print("CardSum: ",self.cards_sum, "Goal: ", self.goal)

# env = Addition()
# obs = env.reset()
# n_steps = 20
# for step in range(n_steps):
#     action = np.random.randint(1,13)
#     obs, reward, done, info = env.step(action)
#     print('obs=', obs, 'reward=', reward, 'done=', done)
#     env.render(mode='human')
#     if done:
#         print("Goal !!", "reward=", reward)
#         break

env = Addition()
obs = env.reset()
n_steps = 20
window_length = 1
input_shape =  (window_length,) + env.observation_space.shape
print("-Initial parameter-")
print(env.action_space) # input
print(env.observation_space) # output
print(env.reward_range) # rewards
print(env.action_space) # action
print(env.action_space.sample()) # action

nb_actions = env.action_space.n
c = input_data = Input(input_shape)
c = Flatten()(c)
c = Dense(128, activation='relu')(c)
c = Dense(128, activation='relu')(c)
c = Dense(128, activation='relu')(c)
c = Dense(128, activation='relu')(c)
c = Dense(nb_actions, activation='linear')(c)
model = Model(input_data, c)
print(model.summary())

# rl
memory = SequentialMemory(limit=50000, window_length=window_length)
policy = EpsGreedyQPolicy() #GreedyQPolicy()# SoftmaxPolicy()
# agent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100, target_model_update=1e-2, policy=policy)
# agent.compile(Adam())
# agent.fit(env, nb_steps=30000, visualize=False, verbose=1)
# agent.save_weights("weights.hdf5")

# predict
model.load_weights("weights_3600000.hdf5")
obs = env.reset()
n_steps = 20
for step in range(n_steps):
    obs = obs.reshape((1, 1, 2))
    action = model.predict(obs)
    # print("predict_action: ", action)
    action = np.argmax(action)
    print("Step {}".format(step + 1))
    print("Action: ", action)
    obs, reward, done, info = env.step(action)
    print('obs=', obs, 'reward=', reward, 'done=', done, '\n')

    env.render(mode='human')

    if done:
        print("Goal !!", "reward=", reward)
        break
