
# %%

import sty
import framework_utils
from generate_datasets.generators.extension_generators import *
from torch.utils.data import DataLoader
import os
from datasets import add_compute_stats
import torch.nn as nn
from experiments.transformations.utils.misc import classes_sets, exp_categorization_task,get_fulltrain_strings_shapenet
from experiments.transformations.utils.datasets import SubclassImageFolder, add_unpack_transf_info, get_transforms
from experiments.transformations.transf_exp import get_weighted_sampler, get_name_samples_file_train, get_name_samples_file_test


def get_exp(transformations, max_objs):
    num_v = 50 if 'v' in transformations else 1
    shared_args = lambda: dict(seed=1, project_name='Transformations', use_weblog=0, batch_size=64, max_epochs=-1 if torch.cuda.is_available() else 2, num_viewpoints_train=num_v, use_mat=0, max_objs_per_class_train=max_objs)
    exp = exp_categorization_task(**shared_args(),
                                  classes_set='set2',
                                  network_name='vgg11bn',
                                  transformations=transformations,
                                  pretraining=pt,
                                  use_device_num=3,
                                  verbose=False)

    name_folder = f'ShapeNet2DFull{"Nomat" if not exp.use_mat else ""}'
    # this are the stats for the untransformed version, whole dataset
    stats = {'mean': [0.06229676, 0.0607271, 0.05646703], 'std': [0.14454809, 0.14061172, 0.12994126]}

    add_PIL_transforms, add_T_transforms = get_transforms(exp.transformations, info=False)

    train_args = dict(add_PIL_transforms=add_PIL_transforms, add_tensor_transforms=add_T_transforms, save_stats_file=None, num_objects_per_class=exp.max_objs_per_class_train, stats=stats, name_classes=classes_sets[exp.classes_set], num_viewpoints_per_object=exp.num_viewpoints_train, name_generator=f"train {exp.classes_set}", num_image_calculate_mean_std=300 if exp.use_cuda else 100, root=f"./data/ShapeNet2D/{name_folder}/train", save_load_samples_filename=get_name_samples_file_train('train', exp))

    train_dataset = add_compute_stats(SubclassImageFolder)(**train_args)

    training_loader = DataLoader(train_dataset, batch_size=exp.batch_size, shuffle=True, num_workers=8 if exp.use_cuda else 0, timeout=10 if exp.use_cuda else 0, pin_memory=True)

    return exp, training_loader


# %%

pt_transform = 's'
pt_objs = 10
pt_num_v = 50 if 'v' in pt_transform else 1
pt = get_fulltrain_strings_shapenet(pt_transform, pt_objs, pt_num_v, class_set='set1', mat=0, network_name='vgg11bn')
# pt = 'vanilla'
exp, training_loader = get_exp(transformations='', max_objs=1)
exp, testing_loader = get_exp(transformations='t', max_objs=1)

# %%

net = exp.get_net(new_num_classes=10)
net.classifier = nn.Sequential(*list(net.classifier.children())[:-2])
# features = torch.zeros(0, 10)
features = torch.zeros(0, 4096)

next(net.parameters()).is_cuda

# %%

N = 100
print(sty.fg.red + f"Loading dataset from {training_loader.dataset.name_generator}" + sty.rs.fg)

y = torch.empty(0, dtype=int)
for idx, data in enumerate(training_loader):
    if idx % 10 == 9:
        print(idx)
    img, lb, more = data
    lo = more['label_object']
    framework_utils.imshow_batch(img)
    #     break
    #     net.cuda()
    net(img.cuda())
    features = torch.vstack((features.cuda(), net(img.cuda())))
    y = torch.hstack((y, lo))
    if idx == N:
        break

import torch


def cossim_metric(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# %%

y.shape

# %%

from sklearn.neighbors import KNeighborsClassifier

neigh_cossim = KNeighborsClassifier(n_neighbors=1, metric=cossim_metric)
neigh_cossim.fit(features.detach().cpu().numpy(), y.detach().cpu().numpy())

# %%

neigh_eucl = KNeighborsClassifier(n_neighbors=1)
neigh_eucl.fit(features.detach().cpu().numpy(), y.detach().cpu().numpy())

# %%

N_test = 5
i = 0
y = torch.empty(0, dtype=int)
features = torch.zeros(0, 4096)
while (i < N_test):
    for idx, data in enumerate(testing_loader):
        if idx % 10 == 9:
            print(idx)
        img, lb, more = data
        lo = more['label_object']

        framework_utils.imshow_batch(img)
        # asda
        features = torch.vstack((features.cuda(), net(img.cuda())))
        y = torch.hstack((y, lo))
        i += 1

p_eucl = neigh_eucl.predict(features.detach().cpu().numpy())
p_cossim = neigh_cossim.predict(features.detach().cpu().numpy())

print(f'Acc K-nn Cossim: {np.mean([i == j for i, j in zip(p_cossim, y.detach().numpy())])}')
print(f'Acc K-nn Eucld: {np.mean([i == j for i, j in zip(p_eucl, y.detach().numpy())])}')

# %%

features.shape

# %%

p_eucl

# %%

p_cossim

# %%


