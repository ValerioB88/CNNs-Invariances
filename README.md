# Learning Online Visual Invariances for Novel Objects via Supervised and Self-Supervised Training
This document contains necessary info to replicate the results from the paper.

# Requirements
A part from the standard ML libraries (and PyTorch), you may need to run:
```
pip install neptune-client
pip install sty
```

# Datasets
## ShapeNet
You can download the ShapeNet dataset from here [https://drive.google.com/file/d/1XiT5JKBHLtGMLmrsAVIEproLs5m7dpWO/view?usp=sharing](https://drive.google.com/file/d/1UHXFWffwMux1P_8p2adeE2gP4HF46dv4/view?usp=sharing)

This version contains 55 classes, 50 objects per class, each object represented from 80 different viewpoints, for a total of 220000 images.
We have verified in the paper that training on 50 Objects or 500 doesn't make much different, and thus we decided to share a more compact version of the full ShapeNet.
To make things work, put the dataset in `./data/ShapeNet2D` (so that the dataset is gonna be in `./data/ShapeNet2D/ShapeNet2DNomat_50objs/`.
We use 10 classes for training and 10 different classes for computing the cosine similarity and other metrics. This is done automatically in the script. If you want to use a lower number of objects (up to 50) you do not need to change the dataset, but it can be done by command line as explained below. 
The 20 classes are all taken from the folder `./data/ShapeNet2D/ShapeNet2DNomat_50objs/train`. The other folders will be used for a variety of tests.


## ETH-80
This dataset can be downloaded from here `https://github.com/Kai-Xuan/ETH-80`
Clone the repo in `./data` in order to end up with the folder `./data/ETH-80-master`. 
To replicate our conditions, you need to remove the background, and make the objects grayscaled.
To do that, run `./code/experiments/transformations/utils/generateETH80dataset.py`.


# Replicate results
To replicate the results you need to 
1. Train each network for each transformations (and a version without any transformation)
2. Run the 5AFT task for each model
3. Run the cosine similarity analysis for each model
4. Plot the results.


## 1. Train each network
This is done in `./code/experiments/transformations/shapenet_fulltrain.py`.
This script uses ML_framework, a custom experimental framework, which accepts a multitude of parameters to control all aspects of training.

To replicate the paper you can simply call `./code/experiments/transformations/run_pipeline_SN_multiobj_simple`. This will run all the networks with all the transformations for all seeds. This will run 9 (networks) * 3 (seeds) * 7 (transformation conditions) = 189 experiments, which may be too long.
 
To run parallel jobs you can check the `./code/experiments/transformations/run_pipeline_SN_multiobj_parallel`. It uses a `job_manager.py` to send multiple jobs and run them one after another across several gpus (3 in the script). This speeds up the training a lot, but I haven't tested it on other machines.

If you just want to run a single condition (one network on one transformation), you can use something like this:

`python code/shapenet_fulltrain.py --network_name samediffnet -transform v --exp_max_obj 50 --shapenet_folder ShapeNet2DNomat_50objs --seed 1`

`network_name` can be any of the followings: samediffnet resnet18 resnet50 vgg11bn vgg19bn alexnet googlenet densenet201
`tranform` can be any of followings: v t r s b c (viewpoint, translation, rotation, scale, brightness, contrast). 
Do not pass `-transform` if you want to train the network _without_ any transformation.
Even though not used in the paper, you can also use combination of transformations (vt, rsb, etc.).
`exp_max_obj` is the number of objects _per class_ you will get from the dataset. The provided dataset has 50 objects, so you can use any number lower than that. 
Leave `shapenet_folder` unchanged. 

Notice that the script will automatically select the 10 classes used for training. 
It will save the results in `models/transformations/ShapeNet/...`.


## 2. Run the 5AFT Task
This is done through a jupyter notebook in `./code/experiments/transformations/analysis/activation_distance_for_accuracy/nway_test.ipynb`
Follow the notebook instructions.
Notice that during these tests NOVEL classes will be used. 


## 3. Run the cosine similarity analysis for each model
Similarly, this is done in `./code/experiments/transformations/activation_distance/run_distances_fulltrain.ipynb`
Notice that during these tests NOVEL classes will be used. 


## 4. Plot the results
After having ran the notebooks, you should have the files in the right places. Then just run the following scripts:

Figure 4 (barplot of 5AFT)
`./code/experiments/transformations/analysis/nway_plot.py`

Figure 5 (invariance analysis)
`./code/experiments/transformations/analysis/invariance_analysis_plot.py`

Figure 6 (multi objects plot)
`./code/experiments/transformations/analysis/multi_objs_plot.py`

Figure 7
_TOP_
`./code/experiments/transformations/analysis/nway_plot.py`
To compute the nway_plot for ETH (after having run the appropriate notebook), you need to change the `dataset` variable to `ETH_grayscale_darkened_nobkg` 
_BOTTOM_
`code/experiments/transformations/analysis/nway_plot_diff.py`


# Neptune.ai Tracking
Across the codebase, I make extensive use of Neptune.ai for data visualization, debug, showing the dataset samples etc. 
However, Neptune.ai is not so commonly used so I disabled it in the provided code. 
If you want to enable, you need to: 
1. Create a Neptune.ai project with name All-Transformations
2. In `./code/ML_framework/experiment.py`, line `192`, change the `neptune_run` with your username.
3. Run shapenet_fulltrain with `-weblog 2`.
