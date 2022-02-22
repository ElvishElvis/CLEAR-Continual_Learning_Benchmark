# The CLEAR Benchmark: Continual LEArning on Real-World Imagery

<img src="https://clear-benchmark.github.io/img/cmu_03.png" alt="drawing" style="width:400px;"/>

<a href="https://linzhiqiu.github.io/">Zhiqiu Lin</a><sup><font color="#A9A9A9">1</font></sup>,
<a href="https://www.linkedin.com/in/elvishelvisshi/">Jia Shi</a><sup><font color="#A9A9A9">1</font></sup>,
<a href="https://www.cs.cmu.edu/~dpathak/">Deepak Pathak*</a><sup><font color="#A9A9A9">1</font></sup>,
<a href="http://www.cs.cmu.edu/~deva/">Deva Ramanan*</a><sup><font color="#A9A9A9">1,2</font></sup>

<sup>1</sup>Carnegie Mellon University 
<sup>2</sup>Argo AI


<a href="https://arxiv.org/abs/2201.06289"><strong>Link to paper</strong></a>: (NeurIPS 2021 Datasets and Benchmarks Track)

<a href="https://clear-benchmark.github.io/"><strong>Link to project page</strong></a>: https://clear-benchmark.github.io/ (include dataset download link)

![](https://clear-benchmark.github.io/img/examples.png)
Continual learning (CL) is widely regarded as crucial challenge for lifelong AI. However, existing CL benchmarks, e.g. Permuted-MNIST and Split-CIFAR, make use of artificial temporal variation and do not align with or generalize to the real-world. In this paper, we introduce CLEAR, the first continual image classification benchmark dataset with a natural temporal evolution of visual concepts in the real world that spans a decade (2004-2014). We build CLEAR from existing large-scale image collections (YFCC100M) through a novel and scalable low-cost approach to visio-linguistic dataset curation. Our pipeline makes use of pre-trained vision-language models (e.g. CLIP) to interactively build labeled datasets, which are further validated with crowd-sourcing to remove errors and even inappropriate images (hidden in original YFCC100M). The major strength of CLEAR over prior CL benchmarks is the smooth temporal evolution of visual concepts with real-world imagery, including both high-quality labeled data along with abundant unlabeled samples per time period for continual semi-supervised learning. We find that a simple unsupervised pre-training step can already boost state-of-the-art CL algorithms that only utilize fully-supervised data. Our analysis also reveals that mainstream CL evaluation protocols that train and test on iid data artificially inflate performance of CL system. To address this, we propose novel "streaming" protocols for CL that always test on the (near) future. Interestingly, streaming protocols (a) can simplify dataset curation since today’s test-set can be repurposed for tomorrow’s train-set and (b) can produce more generalizable models with more accurate estimates of performance since all labeled data from each time-period is used for both training and testing (unlike classic iid train-test splits).

# Codebase 
This repo contain codebase for all classification experiments in our paper. 

### configuration: 
<b>data_folder_path :</b> path for all image (train+test)

<b>data_train_path :</b> path for all training image

<b>data_test_path :</b> path for all testing image

If both data_train_path and data_test_path are provided, it will overwrite data_folder_path and use data_train_path and data_test_path as train/test input path. If not, it will auto train/test split data_folder_path as ratio in test_split(default 0.3)

<b>feature_path :</b> root path for pre-train image feature

<b>split:</b> experiment name, don't affect the program function

<b>load_prev:</b> whether to restore the experiment from previous bucket

<b>image_train</b> and <b>feature_train</b> are use only when running experiment on image directly or on pre-train image feature. These two setting are mutually exclusive(which mean you only need to specify one of them accordingly. 

<b>max_memory_size</b> Maximum number of instance store in the buffer(with replay base method/ reservoir/ bias reservoir). Default buffer size is set to the number of instance in one bucket of timestamp.

<b>num_instance_each_class</b> and <b>num_instance_each_class_test</b>: number of instance in each class in each bucket, randomly removed extra instance from training if there's more instance in the specified folder

<b>random_seed</b>: random seed for experiment(like train/test split, randomly sampling...). Often for testify metric robustness by averaging results from different random seed experiments.

### Training:
Specify <b>pretrain_feature</b> under <b>feature_train</b> setting will automatically parse image feature into <b>feature_path </b>, if not exists. 

<b>pretrain_feature</b> naming convention : prefix(for differentiate setting), pre-train model dataset, pre-train model architecture, dataset name, version of clear dataset, and end with 'feature'. For instance, moco_resnet50_clear_100_feature or test_moco_resnet50_clear_10_feature

For training experiment, run 
```
  python train.sh --yaml
```
An example would be: 

```
  python train.sh clear100/clear100_feature_resnet50_moco.yaml
```
For parsing metric, run
```
python parse_log_to_result.py --split --verbose[to print out the result matrix as well] --move[move to main server to plot] 
```
An example would be: 

```
python parse_log_to_result.py --split clear100_feature_resnet50_moco --verbose 1 --move 1
```
For plotting the result matrix, like one in our paper, first need to specify --move 1 in running parse_log_to_result.py, and then run
```
python get_metric_all.py --plot 1
```

# Contact

Please contact clearbenchmark@gmail.com with any question. Please also follow our website https://clear-benchmark.github.io/ for latest update. 


