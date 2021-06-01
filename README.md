# Learning Online Visual Invariances for Novel Objects via Supervised and Self-Supervised Training
This document contains necessary info to replicate the results from the paper.

# Datasets
## ShapeNet
The ShapeNet dataset with 50 objects per class is included. We have verified in the paper that training on 50 Objects or 500 doesn't make much different, and sharing a ShapeNet dataset with >50 objects is too demanding in terms of space.
To make things work, put the dataset in `./data`
We use 10 classes for training and 10 classes for computing the cosine similarity and other metrics. This is done automatically in the script. You can also use a different number of objects (lower than 50), which will be automatically computed in the script. You won't need to touch this dataset folders. 
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
This script uses ML_framework, my experimental framework, which accepts a multitude of parameters to control all aspects of training.


To replicate the paper (with the ShapeNet x 50 objs) you can simply call `run_pipeline_SN_multiobj_simple`. This will run all the networks with all the transformations for all seeds. However, this may take too long and is not ideal. 
To run parallel jobs you can check the `./code/experiments/transformations/run_pipeline_SN_multiobj_parallel`. I use a `job_manager.py` to send multiple jobs and run them one after another across several gpus (3 in the script). This speeds up the training a lot, but I haven't tested it on other machines.

If you just want to try to run it with one network, you can use something like this:

`python code/shapenet_fulltrain.py --network_name samediffnet -transform v --exp_max_obj 50 --shapenet_folder ShapeNet2DNomat_50objs --seed 1`

`network_name` can be any of the followings: samediffnet resnet18 resnet50 vgg11bn vgg19bn alexnet googlenet densenet201
`tranform` can be any of followings: v t r s b c (viewpoint, translation, rotation, scale, brightness, contrast). 
Do not pass `-transform` if you want to train the network _without_ any transformation.
Even though not used in the paper, you can also use combination of transformations (vt, rsb, etc.).
`exp_max_obj` is the number of objects _per class_ you will get from the dataset. The provided dataset has 50 objects, so you can use any number lower than that. 

Notice that the script will automatically select the 10 classes used for training. 
It will save the results in `models/transformations/ShapeNet/etc`.


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
Across the codebase, I make extensive use of Neptune.ai for data visualization, debug, showing the database etc. 
However, Neptune.ai is not so spreadly used so I disabled it in the provided code. 
If you want to enable, you need to: 
1. Create a Neptune.ai project with name All-Transformations
2. In `./code/ML_framework/experiment.py`, line `192`, change the `neptune_run` with your username.
3. Run shapenet_fulltrain with `-weblog 2`.
