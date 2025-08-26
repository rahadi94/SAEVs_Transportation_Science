import pandas as pd
import numpy as np
import random

'''q_table = dict()
for i in range(10):
    for ii in range(24):
        for iii in range(89):
            for iiii in range(50):
                for iiiii in range(50):
                    state = dict(SOC=i, time=ii, position=iii, supply=iiii, waiting_list=iiiii)
                    q_table[state['SOC'], state['time'], state['position'],
                                 state['supply'], state['waiting_list']] = [round(np.random.uniform(-1, 0), 2)
                                                                            for i in range(2)]

df = pd.DataFrame(q_table)
df.to_csv('q_table.csv')'''
SOC = range(10)
time = range(24)
position = range(89)
supply = range(50)
waiting_list = range(50)

index = pd.MultiIndex.from_product([SOC, time, position, supply, waiting_list],
                                   names=['SOC', 'time', 'position', 'supply', 'waiting_list'])
df = pd.DataFrame(-np.random.rand(len(index), 2), index=index)


