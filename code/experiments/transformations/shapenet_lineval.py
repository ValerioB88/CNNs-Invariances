import argparse
from experiments.transformations.transf_exp import cat_exp
import os
import torch
from sty import fg, bg, ef, rs
from experiments.transformations.utils.misc import project_path, get_fulltrain_strings_shapenet
name_exp_set = 'lineval'

# subfolders = os.path.dirname(globals().get("__file__")).split('code/experiments/')[1]
subfolders = project_path.split("code/experiments/")[1]
model_folder = lambda: f'./models/{subfolders}/ShapeNet/{network_name}/{name_exp_set}'
# model_folder_fulltrain = lambda: f'./models/{subfolders}/ShapeNet/{network_name}/fulltrain/'
test_output_folder = lambda: f'./results/{subfolders}//ShapeNet/{network_name}/{name_exp_set}/'

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("-test_transform", "--test_transforms",
                    help='Transform to apply ONLY DURING TEST. Choose between any combination of vrst e.g. [vf], [rs], [s]. Leave empty for no transformation (only cat)',
                    default='', type=str)
parser.add_argument("-pt_transform", "--pt_transforms",
                    help='',
                    default='', type=str)
parser.add_argument("-pt_objs", "--pt_objs",
                    help='',
                    default=10, type=int)
parser.add_argument("-network_name", "--network_name",
                    default='vgg11bn', type=str)



PARAMS = vars(parser.parse_known_args()[0])
test_transf = PARAMS['test_transforms']
pt_transform = PARAMS['pt_transforms']
pt_objs = PARAMS['pt_objs']
network_name = PARAMS['network_name']

shared_args = lambda: dict(seed=seed, project_name='All-Transformations', use_weblog=2 if torch.cuda.is_available() else False, batch_size=64, max_epochs=-1 if torch.cuda.is_available() else 1, patience_stagnation=500, freeze_all_but_last=freeze_all_but_last, freeze_backbone=freeze_backbone,  pretraining_backbone=ptbkbn, num_viewpoints_train=num_v, use_mat=mat, max_objs_per_class_train=max_objs, add_cat_module=add_cat_module)


def call_cat_exp():
    other_info, output_model, output_test = generate_current_exp_strings()
    exp = cat_exp(**shared_args(),
                  additional_tags=other_info,
                  model_output_filename=output_model,
                  output_filename=output_test,
                  classes_set=classes_set,
                  network_name=network_name,
                  transformations='',
                  transformations_test=test_transf,
                  pretraining=pt,
                  scramble_fc=scramble_fc)
    return exp, output_model, output_test


def generate_current_exp_strings():
    # other_info = f'{exp}_{ptInfo}T{current_exp_transformation}_objs{max_objs}_vp{num_v}_mat{mat}'
    ptInfo = 'vanilla' if pt == 'vanilla' else f'pt[T{pt_transform}+{pt_class_set}+objs{pt_objs}]'
    curr_exp_info = f'T{train_transform}-{test_transf}_objs{max_objs}_vp{num_v}_mat{mat}_tr{classes_set}'
    additional_tags = f'{name_exp_set.upper()}_' + ('vanilla' if pt == 'vanilla' else f'ptT{pt_transform}_ptO{pt_objs}_pt{pt_class_set}') + f'_{curr_exp_info}'

    mid_folder = f'scramble_fc{scramble_fc}' if scramble_fc != 0 else ('add_cat_module' if add_cat_module
                 else ('freeze_all_but_last' if freeze_all_but_last
                 else ('freeze_backbone' if freeze_backbone
                 else ('vanilla' if pt == 'vanilla'
                 else ''))))

    id_string_tr = f"{name_exp_set.upper()}_{ptInfo}_{curr_exp_info}"
    output_model = f"{model_folder()}/{mid_folder}/{id_string_tr}.pt"
    output_test = f'{test_output_folder()}/{mid_folder}/{id_string_tr}_test.pickle'
    return additional_tags, output_model, output_test

ptbkbn = 'vanilla'
mat = 0
seed = 1

###############################################################
## Lineval
train_transform = ''
pt_class_set = 'set1'
pt_num_v = 50 if 'v' in pt_transform else 1
freeze_backbone = 0
freeze_all_but_last = 0
scramble_fc = 0
add_cat_module = 0

num_v = 1
max_objs = 1
classes_set = 'set1'
#
ptbkbn, _ , _, _= get_fulltrain_strings_shapenet(pt_transform, pt_objs, pt_num_v, pt_class_set, mat, 'relationNet')
# ptbkbn = 'vanilla'
pt = 'vanilla'
# network_name = 'vgg11bn'
freeze_backbone = 1
# scramble_fc = 1
exp, output_model, _ = call_cat_exp()


# _, pt, _ = pretraining_strings()
# freeze_all_but_last = 1
# add_cat_module = 1
# exp, output_model, _ = call_cat_exp()


# pt = get_pretraining_strings(pt_transform, pt_objs, pt_num_v, pt_class_set, mat, network_name)
# freeze_backbone = 0
# max_objs=500
# freeze_all_but_last = 1
# exp, output_model, _ = call_cat_exp()

# freeze_backbone = 1
# freeze_all_but_last = 0  # use whatever it's passed as command argument
# exp, output_model, _ = call_cat_exp()


# pt = 'vanilla'
# max_objs = 500
# add_cat_module = 1
# exp, output_model, _ = call_cat_exp()




#
# pt = f'./models/ptImageNet/{network_name}.pt'
# freeze_backbone = 1  # use whatever it's passed as command argument
# exp, output_model, _ = call_cat_exp()
#
#
#
# print(ef.inverse + "Finished" + rs.inverse)

##

