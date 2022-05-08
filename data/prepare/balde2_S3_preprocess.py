import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import torch

color = ['red', 'orange', 'blue', 'green', 'fuchsia', 'black', 'yellow']

inspection_data = sio.loadmat('./blade2_S3_View.mat')['View'][:, :2]

distance = 0.1
resample_data = []
idx = 0
for num in range(inspection_data.shape[0]):
    i = 0
    while np.sqrt(((inspection_data[idx+i]-inspection_data[idx])**2).sum()) < distance:
        i = i+1
        if idx+i == inspection_data.shape[0]-1:
            break

    resample_data.append(inspection_data[idx])
    if idx+i == inspection_data.shape[0] - 1:
        break

    idx = idx + i

inspection_data0 = np.stack(resample_data, axis=0)


distance = 0.2

segmented_data = {}
data_list = []
segmented_num = 0
for idx in range(0, inspection_data0.shape[0]-1):

    if idx!=inspection_data0.shape[0]-2:
        if np.sqrt(((inspection_data0[idx+1]-inspection_data0[idx])**2).sum()) < distance:
            data_list.append(inspection_data0[idx][None, ...])
        else:
            data_list.append(inspection_data0[idx][None, ...])
            print(idx)
            segmented_data['segmented_{}'.format(segmented_num)] = np.concatenate(data_list, axis=0)

            plt.scatter(segmented_data['segmented_{}'.format(segmented_num)][:, 0],
                        segmented_data['segmented_{}'.format(segmented_num)][:, 1],
                        s=5, facecolors='none', edgecolors=color[segmented_num])

            segmented_num = segmented_num + 1
            data_list = []

    else:
        data_list.append(inspection_data0[idx][None, ...])
        data_list.append(inspection_data0[idx+1][None, ...])
        segmented_data['segmented_{}'.format(segmented_num)] = np.concatenate(data_list, axis=0)

        plt.scatter(segmented_data['segmented_{}'.format(segmented_num)][:, 0],
                    segmented_data['segmented_{}'.format(segmented_num)][:, 1],
                    s=5, facecolors='none', edgecolors=color[segmented_num])

plt.show()

segmented_data0 = segmented_data['segmented_0']
segmented_data1 = segmented_data['segmented_1']
segmented_data2 = segmented_data['segmented_2']

plt.scatter(segmented_data0[:, 0],
            segmented_data0[:, 1],
            s=2, facecolors='none', edgecolors=color[0])

plt.scatter(segmented_data1[:, 0],
            segmented_data1[:, 1],
            s=10, facecolors='none', edgecolors=color[1])

plt.scatter(segmented_data2[:, 0],
            segmented_data2[:, 1],
            s=2, facecolors='none', edgecolors=color[2])

plt.xticks([])
plt.yticks([])
# # plt.axis('off')
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98, hspace=0, wspace=0)
plt.show()

all_Src = []
all_Ref = []

src = segmented_data0[-40:-40+32]
index = ((src[:, None, :].repeat(segmented_data1.shape[0], 1)-segmented_data1[None, ...])**2).sum(-1).argmin(axis=-1)
ref = segmented_data1[index[0]:index[0]+32]
plt.scatter(src[:, 0], src[:, 1], s=2, facecolors='none', edgecolors=color[0])
plt.scatter(ref[:, 0], ref[:, 1], s=2, facecolors='none', edgecolors=color[1])
plt.show()
all_Src.append(src[None, ...])
all_Ref.append(ref[None, ...])

src = segmented_data1[120-32:120]
index = ((src[:, None, :].repeat(segmented_data2.shape[0], 1)-segmented_data2[None, ...])**2).sum(-1).argmin(axis=-1)
ref = segmented_data2[index[0]:index[0]+32]
plt.scatter(src[:, 0], src[:, 1], s=2, facecolors='none', edgecolors=color[0])
plt.scatter(ref[:, 0], ref[:, 1], s=2, facecolors='none', edgecolors=color[1])
plt.show()
all_Src.append(src[None, ...])
all_Ref.append(ref[None, ...])

all_Src = np.concatenate(all_Src, 0)   # B*N*2
all_Ref = np.concatenate(all_Ref, 0)   # B*N*2

rand_all_Rot_mat = torch.rand(all_Src.shape[0], 2, 2).numpy()   # B*2*2
rand_all_Translation = torch.rand(all_Src.shape[0], 2).numpy()   # B*2
sio.savemat('./blade2_S3_inference_data_test0.mat',
            {'Src': all_Src,
             'Ref': all_Ref,
             'Rot_mat': rand_all_Rot_mat,
             'Translation': rand_all_Translation})


Inference_data = inspection_data
plt.scatter(Inference_data[:, 0], Inference_data[:, 1],
            s=5, facecolors='none', edgecolors=color[0])
plt.show()
sio.savemat('./blade2_S3_preprocess.mat',
            {'inspection_data': Inference_data})

distance = 0.2
all_Segmented_idx = []

segmented_data = {}
data_list = []
segmented_num = 0
all_Segmented_idx.append(0)
for idx in range(0, Inference_data.shape[0]-1):

    if idx!=Inference_data.shape[0]-2:
        if np.sqrt(((Inference_data[idx+1]-Inference_data[idx])**2).sum()) < distance:
            data_list.append(Inference_data[idx][None, ...])
        else:
            data_list.append(Inference_data[idx][None, ...])
            print(idx)
            all_Segmented_idx.append(idx+1)
            segmented_data['segmented_{}'.format(segmented_num)] = np.concatenate(data_list, axis=0)

            plt.scatter(segmented_data['segmented_{}'.format(segmented_num)][:, 0],
                        segmented_data['segmented_{}'.format(segmented_num)][:, 1],
                        s=5, facecolors='none', edgecolors=color[segmented_num])

            segmented_num = segmented_num + 1
            data_list = []


    else:
        data_list.append(Inference_data[idx][None, ...])
        data_list.append(Inference_data[idx+1][None, ...])
        segmented_data['segmented_{}'.format(segmented_num)] = np.concatenate(data_list, axis=0)

        plt.scatter(segmented_data['segmented_{}'.format(segmented_num)][:, 0],
                    segmented_data['segmented_{}'.format(segmented_num)][:, 1],
                    s=5, facecolors='none', edgecolors=color[segmented_num])

        all_Segmented_idx.append(idx+1+1)

plt.xticks([])
plt.yticks([])
# # plt.axis('off')
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98, hspace=0, wspace=0)
plt.show()

all_Segmented_idx.append(0)
all_Segmented_idx = np.stack(all_Segmented_idx, 0)
sio.savemat('./blade2_S3_inference_data_Segmented_idx.mat',
            {'all_Segmented_idx': all_Segmented_idx})