"""
REORGANIZE THE SHAPENET2D DATASET TO MAKE IT EASY TO TRAIN WITH PYTORCHVISION
"""
import numpy as np
import pathlib
import os
import glob
import shutil
import re
import argparse
import nltk
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet

np.random.seed(1)

syns = list(wordnet.all_synsets())
offsets_list = [(s.offset(), s) for s in syns]
offsets_dict = dict(offsets_list)

parser = argparse.ArgumentParser(allow_abbrev=False)

parser.add_argument("-otr_ote", "--perc_objects_train_test_split", type=int, default=80)
parser.add_argument("-objs_per_class_train", "--total_number_objects_per_class_train", type=int, default=10)  # set to -1 to select all objects
parser.add_argument("-vtr_vte", "--perc_viewpoints_train_test_split", type=int, default=80)
parser.add_argument("-o", "--output_folder_name", default='ShapeNet2DFull_nomat')
args = parser.parse_args()
selfsuperv = 0
perc_train_test_split = args.perc_objects_train_test_split
perc_num_viewpoint_obj_train = args.perc_viewpoints_train_test_split
name_dataset = args.output_folder_name
objs_train = args.total_number_objects_per_class_train
name_to_id = {v.name(): '{:08d}'.format(int(k)) for k, v in offsets_list}

folder = './data/ShapeNet2D/'
whole_dataset = '/whole_dataset_mv_nomat'  # input
from_path = folder + '/' + whole_dataset + '/'
shutil.rmtree(folder + '/' + name_dataset) if os.path.exists(folder + '/' + name_dataset) else None
all_classes_ids = os.listdir(from_path)
num_classes = len(all_classes_ids)

train_percentage_objs = perc_train_test_split

for idx, c in enumerate(all_classes_ids):
    imgs = os.listdir(from_path + c)
    name_class = offsets_dict[int(c)].name()
    print(f'class: {idx}/{len(all_classes_ids)}, {name_class}')
    get_obj_num = lambda name: int(name.split("O")[1].split('_')[0])
    objs_num = np.unique(np.sort([get_obj_num(a) for a in imgs]))
    if objs_train == -1:
        tot_objs = len(objs_num)
        num_objs_train = round(tot_objs * train_percentage_objs/100)
        num_objs_test = tot_objs - num_objs_train
        objs_num_selected = objs_num
    else:
        num_objs_train = objs_train
        num_objs_test = round(objs_train * (100 - train_percentage_objs) / train_percentage_objs)
        tot_objs = num_objs_test + num_objs_train
        try:
            objs_num_selected = np.random.choice(objs_num, tot_objs, replace=False)
        except ValueError:
            print(f"Class {name_class} does not contain enough objects for train and test ({tot_objs}). Skipped")
            continue
    objs_selected_train = np.random.choice(objs_num_selected, num_objs_train, replace=False)
    objs_selected_test = set(objs_num_selected) - set(objs_selected_train)

    # train X classes Y objects, test X classes W objects, test Z classes
    def save_objs(selected_obj_num, type='train', separate_viewpoints=True):
        repetition_obj = [i for i in imgs if re.match(rf"O{selected_obj_num}_", i)]
        if separate_viewpoints:
            viewpoint_obj_train = np.random.choice(repetition_obj, round(len(repetition_obj)*perc_num_viewpoint_obj_train/100), replace=False)
            viewpoint_obj_test = list(set(repetition_obj) - set(viewpoint_obj_train))
            to_path_output = folder + '/' + name_dataset + "/" + type + "/" + name_class + "/"
            save_viewpoint_obj(viewpoint_obj_train, to_path_output)
            to_path_test = folder + '/' + name_dataset + "/testSameObjsDiffViewpoints/" + name_class + "/"
            save_viewpoint_obj(viewpoint_obj_test, to_path_test)
        else:
            to_path_output = folder + '/' + name_dataset + "/" + type + "/" + name_class + "/"
            save_viewpoint_obj(repetition_obj, to_path_output)

    # for each viewpoint of that object
    def save_viewpoint_obj(viewpoint_obj_names, to_folder):
        for idx, v in enumerate(viewpoint_obj_names):
            from_path_obj = from_path + '/' + c + '/' + v
            pathlib.Path(to_folder).mkdir(parents=True, exist_ok=True)
            to_path_obj = to_folder + v
            shutil.copy(from_path_obj, to_path_obj)

    print(f"Train set {len(objs_selected_train)}'")
    for idx, o in enumerate(objs_selected_train):
        if idx % 200 == 0:
            print(f'objects: {idx}/{len(objs_selected_train)}')
        save_objs(o, type='train')

    print(f"Test set {len(objs_selected_test)}")
    for idx, o in enumerate(objs_selected_test):
        if idx % 200 == 0:
            print(f'objects: {idx}/{len(objs_selected_test)}')
        save_objs(o, type='testDiffObjects', separate_viewpoints=False)

##

