from experiments.transformations.cifar10_fulltrain import cat_exp_cifar10
import torch
import argparse

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("-transforms", "--transforms",
                    help='Choose between any combination of vrst e.g. [vf], [rs], [s]. Leave empty for no transformation (only cat)',
                    default='', type=str)
parser.add_argument("-num_imgs", "--num_imgs",
                    default=None, type=int)

seed = 1
freeze_backbone = False
PARAMS = vars(parser.parse_known_args()[0])
exp_transf = PARAMS['transforms']
num_imgs = PARAMS['num_imgs']
network_name = 'allconvC'

shared_args = lambda: dict(seed=seed, project_name='Transformations', use_weblog=2 if torch.cuda.is_available() else 2, batch_size=64, freeze_backbone=freeze_backbone, max_objs_per_class_train=num_imgs)

cat_exp_cifar10(**shared_args(),
                additional_tags=f'CIFAR10_T{exp_transf}_imgs{num_imgs}_{network_name}',
                model_output_filename=f'./models/transformations/cifar10/{network_name}/cifar10_lineval_ptT{exp_transf}_imgs{num_imgs}_{network_name}_Ttest.pt',
                output_filename=f'./results/transformations/cifar10/{network_name}/cifar10_lineval_ptT{exp_transf}_imgs{num_imgs}_{network_name}_Ttest_test.pickle',
                network_name=network_name,
                transformations='',
                transformations_test=exp_transf,
                pretraining=f'./models/transformations/cifar10/{network_name}/cifar10_T{exp_transf}_imgs{num_imgs}_{network_name}.pt',
                max_epochs=350,
                train_or_test='test')
