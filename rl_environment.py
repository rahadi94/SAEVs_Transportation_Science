import gym
from gym import error, spaces, utils
import numpy as np
import logging
import pandas as pd


K = 4

class ChargingHubInvestmentEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    reward_range = (-float('inf'), float('inf'))
    spec = None


    def __init__(self, config, fleet, envv, DQN = False):
        # Set these in ALL subclasses
        self.DQN = DQN
        if DQN == True:
            self.action_space = spaces.Discrete(K**10)
        else:
            self.action_space = spaces.Box(low=0, high=5, shape=(200,), dtype=np.uint8)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=255, shape=
        (1 + 200*3 + 16 + 10, 1), dtype=np.uint8)
        self.fleet = fleet
        self.envv = envv
        self.episode = 0
        # vehicles_to_decide = [vehicle for vehicle in self.fleet.vehicles if vehicle.mode in ['idle','parking','circling']][0:10]
        self.state = self.get_state(self.fleet, self.envv)
        self.current_step = 0
        self.reward = 0
        self.results = np.ndarray((9,0))
        self.env = 'env'
        self._max_episode_steps = 500
        self.config = config
        self.evaluation = config.evaluation

        self.action = None

    def get_state(self, fleet, env):
        vehicles = fleet.vehicles
        charging_stations = fleet.charging_stations
        for j in range(0, 24):
            if j * 60 <= env.now % 1440 <= (j + 1) * 60:
                hour = j
        state = np.array(hour)
        for vehicle in vehicles:
            SOC = int((vehicle.charge_state - vehicle.charge_state % 10) / 10)
            if vehicle.mode in ['idle', 'parking', 'cruising']:
                mode = 1
            else:
                mode = 0
            state = np.append(state, np.array([SOC, vehicle.position.id, mode]))
        q = []
        for i in charging_stations:
            q.append(len(i.plugs.queue) + i.plugs.count)
        q = np.array(q)
        # print(q)
        V2D = []
        assert len(fleet.vehicles_to_decide) == 10, f"we have a problem {fleet.vehicles_to_decide}"
        for i in fleet.vehicles_to_decide:
            V2D.append(i.id)
        V2D = np.array(V2D)
        state = np.append(state, q)
        state = np.append(state, V2D)
        return state

    def step(self, action, fleet, env):
        # Execute one time step within the environment
        reward = self._take_action(action, fleet, env)
        self.current_step += 1
        done = self.current_step >= 100000000000000
        obs = self._next_observation()
        return obs, reward, done, {}

    def receive_action(self):
        return self.action

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        self.reward = 0
        # self.state = self.get_state()
        pd.DataFrame(self.results).to_csv("file.csv")

        return self.get_state(self.fleet, self.envv)


    def render(self, mode='human', close=False):
        print(self.reward)

    def _take_action(self, action, fleet, env):
        state = self.get_state(fleet, env)
        # state = state.reshape((1, self._state_size))
        # lg.info(f'old_state={fleet.old_state}, old_action={fleet.old_action}')
        # lg.info(f'new_action={action}, new_state={state}, {fleet.charging_count}')
        reward = 0

        for vehicle in fleet.vehicles:
            # TODO: fix it
            reward += vehicle.reward['revenue']
            reward += -vehicle.reward['charging']
            reward += -vehicle.reward['queue']
            reward += -vehicle.reward['distance']

            vehicle.old_location = vehicle.location
        reward += -fleet.reward['missed']
        fleet.old_state = state
        fleet.old_action = action
        for vehicle in fleet.vehicles:
            fleet.reward['missed'] = 0
            vehicle.reward['revenue'] = 0
            vehicle.reward['distance'] = 0
            vehicle.reward['charging'] = 0
            vehicle.reward['queue'] = 0
            vehicle.reward['parking'] = 0
            vehicle.reward['missed'] = 0
            vehicle.reward['sub_missed'] = 0
            vehicle.reward['discharging'] = 0
            vehicle.reward['interruption'] = 0
            vehicle.reward['penalty'] = 0
        # print(f'reward = {reward}')
        self.state = state
        return reward

    def _next_observation(self):
        return self.state

def convert_to_vector(a, k = K, h = 9):
    # print(a)
    action = np.zeros(10)
    j = 0
    for i in range(10):
        action[i] = int((a - a%(k**(h-j))) / (k**(h-j)))
        a = a%(k**(h-j))
        j += 1
    # print(action)
    return action

def convert_to_scalar(a, k=K):
    # print(a)
    action = 0
    for i in range(10):
        action += (a[i] * (k) ** (9 - i))
    # print(action)
    return action

def multiply(y):
    output=1
    x = [3,4,3,2,3,3]
    for i in range(y):
        output *= x[i]
    return output