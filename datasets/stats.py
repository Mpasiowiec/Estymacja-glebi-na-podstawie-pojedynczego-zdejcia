import numpy as np
from PIL import Image

with open(
    '/content/drive/MyDrive/magisterka/Estymacja-glebi-na-podstawie-pojedynczego-zdejcia/data_split/kitti_paths.txt',
    "r"
    ) as f:
    filenames_kitti = [s.split()[0] for s in f.readlines()]

with open(
    '/content/drive/MyDrive/magisterka/Estymacja-glebi-na-podstawie-pojedynczego-zdejcia/data_split/nyu_2_paths.txt',
    "r"
    ) as f:
    filenames_nyu = [s.split()[0] for s in f.readlines()]

with open(
    '/content/drive/MyDrive/magisterka/Estymacja-glebi-na-podstawie-pojedynczego-zdejcia/data_split/hypersim_test.txt',
    "r"
    ) as f:
    filenames_hypersim_test = [s.split()[0] for s in f.readlines()]

with open(
    '/content/drive/MyDrive/magisterka/Estymacja-glebi-na-podstawie-pojedynczego-zdejcia/data_split/hypersim_train.txt',
    "r"
    ) as f:
    filenames_hypersim_train = [s.split()[0] for s in f.readlines()]

with open(
    '/content/drive/MyDrive/magisterka/Estymacja-glebi-na-podstawie-pojedynczego-zdejcia/data_split/hypersim_val.txt',
    "r"
    ) as f:
    filenames_hypersim_val = [s.split()[0] for s in f.readlines()]

with open(
    '/content/drive/MyDrive/magisterka/Estymacja-glebi-na-podstawie-pojedynczego-zdejcia/data_split/vkitti2_test.txt',
    "r"
    ) as f:
    filenames_vkitti2_test = [s.split()[0] for s in f.readlines()]

with open(
    '/content/drive/MyDrive/magisterka/Estymacja-glebi-na-podstawie-pojedynczego-zdejcia/data_split/vkitti2_train.txt',
    "r"
    ) as f:
    filenames_vkitti2_train = [s.split()[0] for s in f.readlines()]

with open(
    '/content/drive/MyDrive/magisterka/Estymacja-glebi-na-podstawie-pojedynczego-zdejcia/data_split/vkitti2_val.txt',
    "r"
    ) as f:
    filenames_vkitti2_val = [s.split()[0] for s in f.readlines()]

filenames_vkitti2 = filenames_vkitti2_val.copy()
filenames_vkitti2.extend(filenames_vkitti2_train)
filenames_vkitti2.extend(filenames_vkitti2_test)

filenames_hypersim = filenames_hypersim_val.copy()
filenames_hypersim.extend(filenames_hypersim_train)
filenames_hypersim.extend(filenames_hypersim_test)

filenames_mixed = filenames_hypersim.copy()
filenames_mixed.extend(filenames_vkitti2)

rgb_values_nyu = np.asarray([np.append(np.append(np.sum(np.asarray(Image.open(img).getdata())/255., axis=0), np.sum((np.asarray(Image.open(img).getdata())/255.)**2, axis=0)), len(Image.open(img).getdata())) for img in filenames_nyu])
# rgb_values_nyu = [[sum], [sum^2], count]

total_sum = np.sum(rgb_values_nyu, axis=0)
total_mean = total_sum[:3]/total_sum[-1]
total_std = np.sqrt(total_sum[3:6]/total_sum[-1] - total_mean**2)

print('NYU means: ', total_mean, ' stds: ', total_std)

rgb_values_kitti =  np.asarray([np.append(np.append(np.sum(np.asarray(Image.open(img).getdata())/255., axis=0), np.sum((np.asarray(Image.open(img).getdata())/255.)**2, axis=0)), len(Image.open(img).getdata())) for img in filenames_kitti])

total_sum = np.sum(rgb_values_kitti, axis=0)
total_mean = total_sum[:3]/total_sum[-1]
total_std = np.sqrt(total_sum[3:6]/total_sum[-1] - total_mean**2)

print('KITTI means: ', total_mean, ' stds: ', total_std)

rgb_values_vkitti2 = np.asarray([np.append(np.append(np.sum(np.asarray(Image.open(img).getdata())/255., axis=0), np.sum((np.asarray(Image.open(img).getdata())/255.)**2, axis=0)), len(Image.open(img).getdata())) for img in filenames_vkitti2])

total_sum = np.sum(rgb_values_vkitti2, axis=0)
total_mean = total_sum[:3]/total_sum[-1]
total_std = np.sqrt(total_sum[3:6]/total_sum[-1] - total_mean**2)

print('VKITTI means: ', total_mean, ' stds: ', total_std)

rgb_values_hypersim = np.asarray([np.append(np.append(np.sum(np.asarray(Image.open(img).getdata())/255., axis=0), np.sum((np.asarray(Image.open(img).getdata())/255.)**2, axis=0)), len(Image.open(img).getdata())) for img in filenames_hypersim])

total_sum = np.sum(rgb_values_hypersim, axis=0)
total_mean = total_sum[:3]/total_sum[-1]
total_std = np.sqrt(total_sum[3:6]/total_sum[-1] - total_mean**2)

print('HYPERSIM means: ', total_mean, ' stds: ', total_std)

rgb_values_mixed = np.append(rgb_values_hypersim, rgb_values_vkitti2, axis=0)

total_sum = np.sum(rgb_values_mixed, axis=0)
total_mean = total_sum[:3]/total_sum[-1]
total_std = np.sqrt(total_sum[3:6]/total_sum[-1] - total_mean**2)

print('MIXED means: ', total_mean, ' stds: ', total_std)


# NYU         means:  [0.48012177 0.41071795 0.39187136]  stds:  [0.28875302 0.29516797 0.30792887]
# KITTI       means:  [0.38416928 0.4104948  0.38838536]  stds:  [0.30759685 0.31810902 0.32846335]

# VKITTIval   means:  [0.3849854  0.38966277 0.3119897 ]  stds:  [0.27087488 0.2705229  0.27190212]
# VKITTItest  means:  [0.38241672 0.38700944 0.30898638]  stds:  [0.27035659 0.27008061 0.27130597]
# HYPERSIMtestmeans:  [0.43227474 0.40066626 0.36778125]  stds:  [0.36698599 0.36501711 0.36375441]
# HYPERSIMval means:  [0.43323305 0.40123018 0.3691462 ]  stds:  [0.3672226  0.36521122 0.3643544 ]
# MIXEDval    means:  [0.42240857 0.398635   0.356323  ]  stds:  [0.34851759 0.34626185 0.3465912 ]
# MIXEDtest   means:  [0.42114056 0.39761645 0.35465131]  stds:  [0.34836277 0.34612877 0.34612973]