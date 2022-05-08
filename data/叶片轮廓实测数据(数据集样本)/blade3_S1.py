import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TXT_PATH = 'blade3_S1_delete.txt'

raw_data = pd.read_table(TXT_PATH).values

threshold = 0.03
segmented_data = {}
data_list = []
segmented_num = 0
color = ['red', 'orange', 'blue', 'green', 'fuchsia', 'black', 'yellow']
discount_factor = 0.5
for idx in range(0, raw_data.shape[0]-1):

    if idx!=raw_data.shape[0]-2:
        if np.abs(raw_data[idx + 1][1] - raw_data[idx][1]) < threshold:
            data_list.append(raw_data[idx][None, ...])
        else:
            data_list.append(raw_data[idx][None, ...])
            print(idx)
            segmented_data['segmented{}'.format(segmented_num)] = np.concatenate(data_list, axis=0)

            plt.scatter(segmented_data['segmented{}'.format(segmented_num)][:, 0],
                        segmented_data['segmented{}'.format(segmented_num)][:, 1],
                        s=discount_factor**segmented_num,
                        c=color[segmented_num])

            segmented_num = segmented_num + 1
            data_list = []

    else:
        data_list.append(raw_data[idx][None, ...])
        data_list.append(raw_data[idx+1][None, ...])
        segmented_data['segmented{}'.format(segmented_num)] = np.concatenate(data_list, axis=0)

        plt.scatter(segmented_data['segmented{}'.format(segmented_num)][:, 0],
                    segmented_data['segmented{}'.format(segmented_num)][:, 1],
                    s=discount_factor**segmented_num,
                    c=color[segmented_num])


plt.show()
