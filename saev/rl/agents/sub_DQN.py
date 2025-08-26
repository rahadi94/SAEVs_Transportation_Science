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
    return epsilon / 10


class sub_Agent:
    def __init__(self, episode):

        # Initialize attributes
        # self.env = env
        self._state_size = 22
        self._action_size = 16
        self._optimizer = Adam(learning_rate=0.0001)
        self.batch_size = 32
        self.expirience_replay = deque(maxlen=100000)
        # self.episode = episode
        # Initialize discount and exploration rate
        self.gamma = 0.99
        self.Gamma = 0.90
        self.episode = episode
        self.counter = 0

        # Build networks
        self.q_network = self._build_compile_model()
        self.target_network = self._build_compile_model()
        self.alighn_target_model()

    def get_state(self, vehicle, charging_stations, vehicles, waiting_list, env):
        SOC = int((vehicle.charge_state - vehicle.charge_state % 10) / 10)
        if isinstance(SOC, np.ndarray):
            SOC = SOC[0]
        for j in range(0, 24):
            if j * 60 <= env.now % 1440 <= (j + 1) * 60:
                hour = j
        position = vehicle.position.id
        supply = len([v for v in vehicles if v.location.distance_1(vehicle.location) <= 4 and v.charge_state >= 30 and
                      v.mode in ['idle', 'parking', 'circling', 'queue']])
        if isinstance(supply, np.ndarray):
            supply = supply[0]
        wl = len([t for t in waiting_list if t.origin.distance_1(vehicle.location) <= 5])
        if isinstance(wl, np.ndarray):
            wl = wl[0]
        q = []
        for i in charging_stations:
            q.append(len(i.plugs.queue) + i.plugs.count)
        q = np.array(q)
        number_free_CS = 0
        for CS in charging_stations:
            if CS.plugs.count < CS.capacity:
                number_free_CS += 1
        if number_free_CS > 1:
            free_CS = 1
        else:
            free_CS = 0
        return np.append(np.array([SOC, hour, position, supply, free_CS, wl]), q)

    def store(self, state, action, reward, next_state, period):
        self.expirience_replay.append((state, action, reward, next_state, period))

    def _build_compile_model(self):
        if self.episode > 0:
            json_file = open('sub_model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            # load weights into new model
            model.load_weights("sub_model.h5")
            # evaluate loaded model on test data
            model.compile(loss='mse', optimizer=self._optimizer)
        else:
            model = Sequential()
            # model.add(Embedding(self._state_size, 10, input_length=1))
            # model.add(Reshape((None,7)))
            model.add(Dense(512, activation='relu', input_dim=self._state_size))
            model.add(Dense(256, activation='relu'))
            model.add(Dense(512, activation='relu'))
            model.add(Dense(self._action_size, activation='linear'))

            model.compile(loss='mse', optimizer=self._optimizer)
        return model

    def alighn_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())

    '''def act(self, state, episode, CSs, vehicle):
        epsilon = epsilon_decay(episode)
        if self.counter <= 500 and episode == 0:
            action = np.random.choice(np.arange(16))
        else:
            available_CS = [CSs.index(x) for x in CSs if len(x.plugs.queue) < 10]
            c1=closest_facility(CSs, vehicle)
            free_CS = [x for x in CSs if x.plugs.count < x.capacity]
            c2=closest_facility(free_CS, vehicle)
            fast_CS = [x for x in CSs if x.power == 50 / 60]
            c3=closest_facility(fast_CS, vehicle)
            available_CS = [CSs.index(x) for x in CSs if x in [c1,c2,c3]]
            if np.random.rand() <= epsilon:
                action = np.random.choice(available_CS)
            else:
                q_values = self.q_network.predict(state)
                df = pd.DataFrame(q_values)
                action = df[available_CS].idxmax(axis=1).values[0]
        return action'''

    def act(self, state, episode, CSs, vehicle):
        epsilon = epsilon_decay(episode)
        if self.counter <= 500 and episode == 0:
            action = np.random.choice(np.arange(16))
        else:
            CSs = [x for x in CSs if len(x.plugs.queue) < 10]
            c1 = closest_facility(CSs, vehicle)
            free_CS = [x for x in CSs if x.plugs.count < x.capacity]
            c2 = closest_facility(free_CS, vehicle)
            fast_CS = [x for x in CSs if x.power == 50 / 60]
            c3 = closest_facility(fast_CS, vehicle)
            available_CS = [CSs.index(x) for x in CSs if x in [c1, c2, c3]]
            if np.random.rand() <= epsilon:
                action = np.random.choice(available_CS)
            else:
                q_values = self.q_network.predict(state)
                df = pd.DataFrame(q_values)
                action = df[available_CS].idxmax(axis=1).values[0]
        return action

    def retrain(self, batch_size, CSs, vehicle):
        minibatch = random.sample(self.expirience_replay, batch_size)

        for state, action, reward, next_state, period in minibatch:

            target = self.q_network.predict(state)
            t = self.target_network.predict(next_state)
            df = pd.DataFrame(t)
            CSs = [x for x in CSs if len(x.plugs.queue) < 10]
            c1 = closest_facility(CSs, vehicle)
            free_CS = [x for x in CSs if x.plugs.count < x.capacity]
            c2 = closest_facility(free_CS, vehicle)
            fast_CS = [x for x in CSs if x.power == 50 / 60]
            c3 = closest_facility(fast_CS, vehicle)
            available_CS = [CSs.index(x) for x in CSs if x in [c1, c2, c3]]
            a = df[available_CS].max(axis=1).values[0]
            k = ceil(period / 15)
            target[0][action] = reward + self.gamma ** k * a

            self.q_network.fit(state, target, epochs=1, verbose=0)

    def take_action(self, vehicle, charging_stations, vehicles, waiting_list, env, episode):
        self.counter += 1
        state = self.get_state(vehicle, charging_stations, vehicles, waiting_list, env)
        state = state.reshape((1, len(state)))
        lg.info(f'old_state={vehicle.old_state}, old_sub_action={vehicle.old_sub_action}')
        action = self.act(state, episode, charging_stations, vehicle)
        vehicle.old_location = vehicle.location
        lg.info(f'new_sub_action={action}, new_state={state}, {vehicle.charging_count}')
        if len(self.expirience_replay) > 512:
            if len(self.expirience_replay) % 10 == 1:
                self.retrain(self.batch_size, charging_stations, vehicle)
            if len(self.expirience_replay) % 1000 == 1:
                self.alighn_target_model()
        vehicle.old_sub_time = env.now
        vehicle.old_sub_action = action
        vehicle.old_sub_state = state
        return action
