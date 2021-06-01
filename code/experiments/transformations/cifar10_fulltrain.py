# %%
import argparse
from generate_datasets.generators.extension_generators import *
from torch.utils.data import DataLoader
import os


from pathlib import Path
from experiments.transformations.transf_exp import exp_categorization_task
from experiments.transformations.utils.datasets import get_transforms
from datasets import MyImageFolder, add_compute_stats, SubsetImageFolder

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
desired_width = 320
np.set_printoptions(linewidth=desired_width)

this_file_path = Path(globals().get("__file__", "./_")).absolute().parent


def cat_exp_cifar10(**experiment_params):
    exp = exp_categorization_task(**experiment_params)
    stats = {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.247, 0.243, 0.261]}
    add_PIL_transforms, add_T_transforms = get_transforms(exp.transformations, info=False)

    train_args = {'add_PIL_transforms': add_PIL_transforms, 'add_tensor_transforms': add_T_transforms, 'save_stats_file': None, 'stats': stats, }
    # add_PIL_transforms += [torchvision.transforms.Resize(size=(128, 128))]
    train_dataset = add_compute_stats(SubsetImageFolder)(root=f'./data/CIFAR10/train',
                                                         num_images_per_class=exp.max_objs_per_class_train,
                                                         save_load_samples_filename=f'./data/CIFAR10/samples/train_{exp.max_objs_per_class_train}_numC{len(cifar10_set1)}.s',
                                                         name_generator="train cifar10",
                                                         name_classes=cifar10_set1,
                                                         **train_args)

    training_loader = DataLoader(train_dataset, batch_size=exp.batch_size, shuffle=True, num_workers=8 if exp.use_cuda else 0, timeout=10 if exp.use_cuda else 0, pin_memory=True)

    ################ TESTING ##################
    dataset_test = []
    dataset_test += [add_compute_stats(MyImageFolder)(root='./data/CIFAR10/test',
                                                      name_generator="test cifar10",
                                                      name_classes=cifar10_set1,
                                                      **train_args)]

    testing_loader_list = ([DataLoader(dataset, batch_size=exp.batch_size, shuffle=True, num_workers=8 if exp.use_cuda else 0, timeout=10 if exp.use_cuda else 0, pin_memory=True) for dataset in dataset_test])

    ###########################################
    try:
        for run in range(exp.num_runs):
            framework_utils.weblogging_plot_generators_info(training_loader, testing_loader_list, exp.weblogger, num_batches_to_log=np.max((1, round(20 / exp.batch_size))))
            if exp.do_train:
                net = exp.train(training_loader, log_text=training_loader.dataset.name_generator, test_loaders=testing_loader_list)
            else:
                net = exp.get_net(new_num_classes=training_loader.dataset.num_classes)
            exp.test(net, testing_loader_list)
            if run < (exp.num_runs - 1):
                exp.new_run()
        exp.save_all_runs_and_stop()
        exp.weblogger['STATUS'] = 'Success'
        return exp
    except:
        exp.weblogger['STATUS'] = 'Failed'
        raise


if __name__ == '__main__':
    # all_cifar10 = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    cifar10_set1 = ['airplane', 'automobile', 'cat', 'deer', 'frog', 'horse', 'truck']
    # cifar10_set2 = ['bird', 'dog', 'ship']
    freeze_support()

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("-transform", "--transform",
                        help='Choose between any combination of vrst e.g. [vf], [rs], [s]. Leave empty for no transformation (only cat)',
                        default='', type=str)
    parser.add_argument("-num_imgs", "--num_imgs",
                        default=None, type=int)
    parser.add_argument("-seed", "--seed",
                        default=1, type=int)

    freeze_backbone = False
    PARAMS = vars(parser.parse_known_args()[0])
    exp_transf = PARAMS['transform']
    num_imgs = PARAMS['num_imgs']
    seed = PARAMS['seed']
    network_name = 'vgg11bn'

    shared_args = lambda: dict(seed=seed, project_name='All-Transformations', use_weblog=2 if torch.cuda.is_available() else 2, freeze_backbone=freeze_backbone)

    cat_exp_cifar10(**shared_args(),
                    additional_tags=f'CIFAR10_T{exp_transf}_imgs{num_imgs}_{network_name}',
                    model_output_filename=f'./models/transformations/cifar10/{network_name}/fulltrain/FULLTRAIN_T{exp_transf}_imgs{num_imgs}_{network_name}_S{seed}.pt',
                    output_filename=f'./results/transformations/cifar10/{network_name}/fulltrain/FULLTRAIN_T{exp_transf}_imgs{num_imgs}_{network_name}_S{seed}_test.pickle',
                    network_name=network_name,
                    transformations=exp_transf,
                    pretraining='vanilla',
                    batch_size=64,
                    max_epochs=-1 if torch.cuda.is_available() else 50,
                    patience_stagnation=-1 if '0' in exp_transf else 500,
                    max_objs_per_class_train=num_imgs)
