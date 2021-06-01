import matplotlib.pyplot as plt
import seaborn as sn
from ML_framework.distance_activation import DistanceActivation
from generate_datasets.generators.extension_generators import *
from torch.utils.data import DataLoader
from datasets import add_compute_stats
from experiments.transformations.utils.misc import classes_sets, exp_categorization_task, get_fulltrain_strings_shapenet
from experiments.transformations.utils.datasets import SubclassImageFolder, get_transforms
##
network_name = 'vgg11bn'
dataset_name = 'ShapeNet'

import torchvision
def change_transformation_dataset(dataset, transformations):
    # num_v = 50 if 'v' in transformations else 1
    add_PIL_transforms, add_T_transforms = get_transforms(transformations, info=False)
    dataset.transform = torchvision.transforms.Compose([*add_PIL_transforms, torchvision.transforms.ToTensor(), *add_T_transforms])
    normalize = torchvision.transforms.Normalize(mean=stats['mean'],
                                                         std=stats['std'])
    dataset.transform.transforms += [normalize]


def get_exp(max_objs):
    shared_args = lambda: dict(seed=1, project_name='Transformations', use_weblog=0, batch_size=64, max_epochs=-1 if torch.cuda.is_available() else 2, num_viewpoints_train=1, use_mat=0, max_objs_per_class_train=max_objs)
    exp = exp_categorization_task(**shared_args(),
                                  classes_set='set2',
                                  network_name=network_name,
                                  use_device_num=3,
                                  verbose=False)

    name_folder = f'ShapeNet2DFull{"Nomat" if not exp.use_mat else ""}'

    add_PIL_transforms, add_T_transforms = get_transforms('', info=False)
    args = dict(name_generator=f"train {exp.classes_set}", add_PIL_transforms=add_PIL_transforms, add_tensor_transforms=add_T_transforms, save_stats_file=None, stats=stats, num_objects_per_class=exp.max_objs_per_class_train, name_classes=classes_sets[exp.classes_set], num_viewpoints_per_object=exp.num_viewpoints_train,root=f"./data/ShapeNet2D/{name_folder}/train", save_load_samples_filename='./train_set2_numO3_numV1_S1.s')
    dataset = add_compute_stats(SubclassImageFolder)(**args)
    loader = DataLoader(dataset, batch_size=exp.batch_size, shuffle=False, num_workers=0, timeout=0, pin_memory=True)

    return exp, dataset, loader
stats = {'mean': [0.06229676, 0.0607271, 0.05646703], 'std': [0.14454809, 0.14061172, 0.12994126]}

def get_net(pt_transform, pt_objs=None):
    pt_num_v = 50 if 'v' in pt_transform else 1

    if pt_transform == 'vanilla':
        exp.pretraining = 'vanilla'
        exp.network_name = network_name
        net = exp.get_net(10)
        net_name = 'vanilla'
        folder_output = 'others'

    elif pt_transform == 'ImageNet':
        exp.pretraining = './models/ptImageNet/vgg11bn.pt'
        net = exp.get_net(1000)
        net_name = 'ImageNet'
        folder_output = 'others'
    else:
        exp.pretraining, net_name = get_fulltrain_strings_shapenet(pt_transform, pt_objs, pt_num_v, class_set='set1', mat=0, network_name='vgg11bn')
        folder_output = 'fulltrain'
        net = exp.get_net(new_num_classes=10)
    # net.classifier = nn.Sequential(*list(net.classifier.children())[:-3])  # before last classifier and dropout
    # features = torch.zeros(0, 10)
    # features = torch.zeros(0, 4096)
    net.eval()
    return net


class SanityCheck(DistanceActivation):
    def get_base_and_other_canvasses(self, class_num, name_class):
        pass


exp, dataset, loader = get_exp(3)
change_transformation_dataset(dataset, 't')
for idx, data in enumerate(loader):
    img_set1, lb, lo = data

for idx, data in enumerate(loader):
    img_set2, lb, lo = data


## EVALUATE IMAGENET
def do_it(pt_transform, pt_obj=None):
    net = get_net(pt_transform, pt_obj)

    sc = SanityCheck(net, distance='cossim')
    i = 0
    all_cossim = np.zeros((30, 30))
    for i in range(30):
        cossim_net, cossim_images,  p = sc.get_cosine_similarity_from_images(img_set1[i], img_set2)
        all_cossim[i, :] = cossim_net[sc.all_layers_name[-2]]

    return all_cossim

all_cossim_im = do_it('ImageNet')

all_cossim_t = do_it('t', 500)

##
plt.close('all')
figure = plt.figure(figsize=(12, 10))
sn.heatmap(all_cossim_im, annot=True, fmt=".2f", annot_kws={"size": 10}, vmin=0, vmax=1)  # font size
np.sum(np.argmax(all_cossim_im, 0) == np.arange(30))
acc = np.sum(np.argmax(all_cossim_im, 0) == np.arange(30))
plt.title(f'IMAGENET acc {acc}/30')
plt.tight_layout()

figure = plt.figure(figsize=(12, 10))
sn.heatmap(all_cossim_t, annot=True, fmt=".2f", annot_kws={"size": 10}, vmin=0, vmax=1)  # font size
np.sum(np.argmax(all_cossim_t, 0) == np.arange(30))
acc = np.sum(np.argmax(all_cossim_t, 0) == np.arange(30))
plt.tight_layout()
plt.title(f'TRANSFORMATIOn acc {acc}/30')
##

