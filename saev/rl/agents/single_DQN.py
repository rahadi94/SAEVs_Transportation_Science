import random
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
import numpy as np
from collections import deque
from saev.core.location import closest_facility
from saev.core.log import lg
import math
import pandas as pd
from math import ceil

A = 0.5
B = 0.1
C = 0.1
EPISODES = 20


def epsilon_decay(time):
    standardized_time = (time - A * EPISODES) / (B * EPISODES)
    cosh = np.cosh(math.exp(-standardized_time))
    epsilon = 1.1 - (1 / cosh + (time * C / EPISODES))
    return epsilon / 5


class SingleAgent:
    def __init__(self, episode):

        # Initialize atributes
        # self.env = env
        self._state_size = 1 + 200*3 + 16 + 10
        self._action_size = 3**10
        self._optimizer = Adam(learning_rate=0.0001)
        self.batch_size = 4
        self.expirience_replay = deque(maxlen=100000)
        # Initialize discount and exploration rate
        self.gamma = 0.99
        self.Gamma = 0.90
        self.replay_start_size = 4
        self.episode = episode
        # Build networks
        self.q_network = self._build_compile_model()
        self.target_network = self._build_compile_model()
        self.alighn_target_model()

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

    def store(self, state, action, reward, next_state):
        self.expirience_replay.append((state, action, reward, next_state))

    def _build_compile_model(self):
        if self.episode > 0:
            json_file = open('model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            # load weights into new model
            model.load_weights("model.h5")
            # evaluate loaded model on test data
            model.compile(loss='mse', optimizer=self._optimizer)
        else:
            model = Sequential()
            # model.add(Embedding(self._state_size, 10, input_length=1))
            # model.add(Reshape((None,7)))
            model.add(Dense(512, activation='relu', input_dim=self._state_size))
            model.add(Dense(256, activation='relu'))
            # model.add(Dense(256, activation='relu'))
            model.add(Dense(512, activation='relu'))
            model.add(Dense(self._action_size, activation='linear'))

            model.compile(loss='mse', optimizer=self._optimizer)
        return model

    def alighn_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def checked_action(self, action, vehicles, state):
        action = self.convert_to_vector(action)
        i = 0
        for vehicle in vehicles:
            if vehicle.mode not in ['idle', 'parking', 'circling']:
                action[i] = 0
            if vehicle.mode in ['idle', 'parking', 'circling'] and vehicle.charge_state < 25:
                action[i] = 2
            if action[i] in [1, 2]:
                if vehicle.charge_state > 90:
                    action[i] = 0
            # if action[i] == 4:
            #     if vehicle.charge_state <= 50 or vehicle.mode not in ['idle', 'parking'] or \
            #             len(vehicle.position.list_of_vehicles) <= vehicle.position.demand.iloc[0, state[0][1]]:
            #         action[i] = 0
            i += 1
        action = self.convert_to_scalar(action)
        return action

    def act(self, state, episode, vehicles):
        epsilon = epsilon_decay(episode)
        if np.random.rand() <= epsilon:
            action = np.random.choice(range(self._action_size))
        else:
            q_values = self.q_network.predict(state)
            df = pd.DataFrame(q_values)
            action = np.argmax(df)
        action = int(self.checked_action(action, vehicles, state))
        return action

    def retrain(self, batch_size):
        minibatch = random.sample(self.expirience_replay, batch_size)

        for state, action, reward, next_state in minibatch:

            target = self.q_network.predict(state)

            t = self.target_network.predict(next_state)
            df = pd.DataFrame(t)
            df = df.loc[0,range(10)]
            k = 1
            target[0][action] = reward + (self.gamma ** k) * np.amax(np.array(df.values))

            self.q_network.fit(state, target, epochs=1, verbose=0)

    def take_action(self, vehicles, env, episode, fleet):
        state = self.get_state(fleet, env)
        state = state.reshape((1, self._state_size))
        lg.info(f'old_state={fleet.old_state}, old_action={fleet.old_action}')
        action = self.act(state, episode, vehicles)
        lg.info(f'new_action={action}, new_state={state}, {fleet.charging_count}')
        reward = 0

        for vehicle in fleet.vehicles:
            # TODO: fix it
            reward += vehicle.reward['revenue']
            reward += -vehicle.reward['charging']
            reward += -vehicle.reward['queue']
            reward += -vehicle.reward['distance']

            vehicle.old_location = vehicle.location

        reward += -fleet.reward['missed']

        if fleet.old_state is not None:
            self.store(fleet.old_state, fleet.old_action, reward, state)
        if len(self.expirience_replay) > self.replay_start_size:
            if len(self.expirience_replay) % 4 == 1:
                self.retrain(self.batch_size)
        if len(self.expirience_replay) % 4 == 1:
            self.alighn_target_model()

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
        return action

    def convert_to_vector(self, a, k=3, h=9):
        # print(a)
        action = np.zeros(10)
        j = 0
        for i in range(10):
            action[i] = int((a - a % (k ** (h - j))) / (k ** (h - j)))
            a = a % (k ** (h - j))
            j += 1
        # print(action)
        return action

    def convert_to_scalar(self, a, k=3):
        # print(a)
        action = 0
        for i in range(10):
            action += (a[i] * (k) ** (9 - i))
        # print(action)
        return action

