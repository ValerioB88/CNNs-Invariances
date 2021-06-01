import argparse
from experiments.transformations.utils.misc import get_fulltrain_strings_shapenet, exp_categorization_task
from experiments.transformations.utils.datasets import SubclassImageFolder, SameDifferentSampler, get_transforms
import os
import torch
from datasets import add_compute_stats
import numpy as np
from torch.utils.data import DataLoader

import framework_utils


def get_sampler_or_batch_sampler(dataset):
    if exp.network_name == 'samediffnet':
        return dict(batch_sampler=SameDifferentSampler(batch_size=exp.batch_size, prob_same=0.5, rebalance_classes=True, subclasses=dataset.subclasses, subclasses_names=dataset.subclasses_names, dataset=dataset))
    else:
        return dict(shuffle=True, batch_size=exp.batch_size)


parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("-pt_transform", "--pt_transform",
                    help='transformation the network is trained on',
                    default='', type=str)

PARAMS = vars(parser.parse_known_args()[0])
pt_transf = PARAMS['pt_transform']

exp = exp_categorization_task(project_name='All-Transformations',
                              use_weblog=2 if torch.cuda.is_available() else 2,
                              batch_size=64)
other_info, output_model, output_test, _ = get_fulltrain_strings_shapenet(pt_transf, exp.max_objs_per_class_train,
                                                                 50 if 'v' in pt_transf else 1, 'set1', 0, exp.network_name,
                                                                 exp.seed)

exp.output_filename = f'./results/transformations/ETH/{exp.network_name}/only_tests/{os.path.splitext(os.path.basename(output_test))[0]}_testT{exp.transformations}.pickle'
exp.weblogger['sys/tags'].add([f'ptT{pt_transf}', f'ptObj{exp.max_objs_per_class_train}', 'ETH', 'NoTomato', 'NoExtremeIncl', exp.network_name])
exp.weblogger['pt_transformation'] = pt_transf
exp.pretraining = output_model
# exp.pretraining = 'vanilla'
stats = {'mean': [0.06229676, 0.0607271, 0.05646703], 'std': [0.14454809, 0.14061172, 0.12994126]}
add_PIL_transforms, add_T_transforms = get_transforms(exp.transformations, info=False)

dataset_test = add_compute_stats(SubclassImageFolder)(name_generator=f'test ETH', root='./data/ETH-80-master/transformed_darkened/', num_objects_per_class=None, stats=stats, num_viewpoints_per_object=50 if 'v' in exp.transformations else 1, add_PIL_transforms=add_PIL_transforms, add_tensor_transforms=add_T_transforms, name_classes=['apple', 'car', 'cow', 'cup', 'dog', 'horse', 'pear'], save_load_samples_filename=None)

testing_loader = DataLoader(dataset_test, **get_sampler_or_batch_sampler(dataset_test), num_workers=8 if exp.use_cuda else 0, timeout=60 if exp.use_cuda else 0, pin_memory=True)


try:
    framework_utils.weblogging_plot_generators_info(None, [testing_loader], exp.weblogger, num_batches_to_log=np.max((1, round(20/exp.batch_size))))
    net = exp.get_net(new_num_classes=testing_loader.dataset.num_classes)
    exp.test(net, [testing_loader])
    exp.save_all_runs_and_stop()
    exp.weblogger['STATUS'] = 'Success'
except:
    exp.weblogger['STATUS'] = 'Failed'
    raise


