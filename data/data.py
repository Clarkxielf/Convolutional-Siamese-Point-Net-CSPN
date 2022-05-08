import scipy.io as sio
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

color = ['red', 'orange', 'blue', 'green', 'fuchsia', 'black', 'yellow']


TXT_PATH = './blade3_S1_delete.txt'

raw_data = pd.read_table(TXT_PATH).values

distance = 0.1
resample_data = []
idx = 0
for num in range(raw_data.shape[0]):
    i = 0
    while np.sqrt(((raw_data[idx+i]-raw_data[idx])**2).sum()) < distance:
        i = i+1
        if idx+i == raw_data.shape[0]-1:
            break

    resample_data.append(raw_data[idx])
    if idx+i == raw_data.shape[0] - 1:
        break

    idx = idx + i

raw_data = np.stack(resample_data, axis=0)

plt.scatter(raw_data[:, 0], raw_data[:, 1], s=5, facecolors='none', edgecolors=color[0])
plt.show()


num_sample = 2000
src_num_sample_points = 32
ref_num_sample_points = 36

all_Src = []
all_Ref = []
all_Rot_mat = []
all_Translation = []
for num in range(num_sample):

    extend_num = np.random.randint(0, ref_num_sample_points - src_num_sample_points, 1)

    base_num = np.random.randint(extend_num, raw_data.shape[0]-ref_num_sample_points+extend_num, 1)
    src = raw_data[int(base_num):int(base_num+src_num_sample_points), :2]
    raw_ref = raw_data[int(base_num-extend_num):int(base_num+ref_num_sample_points-extend_num), :2]

    assert src.shape[0] == src_num_sample_points and raw_ref.shape[0] == ref_num_sample_points


    TheTa = np.random.random(1) * np.pi/36
    Translation = 2 * (np.random.random((1, 2)) - 0.5) * 0.5
    Rot_sr = np.array([[np.cos(TheTa), -np.sin(TheTa)],
                       [np.sin(TheTa), np.cos(TheTa)]]).squeeze()

    ref = raw_ref @ Rot_sr.transpose(-1, -2) + Translation

    # plt.scatter(src[:, 0], src[:, 1],  s=2, facecolors='none', edgecolors=color[0])
    # plt.scatter(raw_ref[:, 0], raw_ref[:, 1],  s=5, facecolors='none', edgecolors=color[1])
    # plt.scatter(ref[:, 0], ref[:, 1],  s=5, facecolors='none', edgecolors=color[2])
    # plt.show()

    all_Src.append(src[None, ...])
    all_Ref.append(ref[None, ...])
    all_Rot_mat.append(Rot_sr[None, ...])
    all_Translation.append(Translation)


all_Src = np.concatenate(all_Src, 0)   # B*N*2
all_Ref = np.concatenate(all_Ref, 0)   # B*N*2
all_Rot_mat = np.concatenate(all_Rot_mat, 0)   # B*2*2
all_Translation = np.concatenate(all_Translation, 0)   # B*2

# sio.savemat('0.1_blade3_32data_2d_test0.mat',
#             {'Src': all_Src,
#              'Ref': all_Ref,
#              'Rot_mat': all_Rot_mat,
#              'Translation': all_Translation})