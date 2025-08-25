## RAPPIE-pytorch

## Introduction

We proposed the RAPPIE model,it contains four core modules: 

1) We utilize reader information prompts to construct the reader agent for feedback simulation. 
2) We construct a global interactive multi-behavior overlapping network is constructed based on the simulated behaviors of reposting and reposting with a comment, as well as following, among different users.
3) We utilize a role-aware multi-view GNN is employed to learn users representations from multiple interactive perspectives, integrating reader propagation role during the graph learning process.
4) We achieve implicit emotion prediction is carried out by fusing content semantic with propagation role features and reader-feedback enhanced author embeddings.


## Enviroment Requirement

`pip install -r requirements.txt`

#### lightGCN

We provide three processed datasets: follow, repost,comment in `lightGCN/weibo_xxx/data/user_f`.

We provide two processed datasets: repost,comment in `lightGCN/twitter_xxx/data/user_f`.

#### RAPPIE

We provide processed datasets: Weibo, Twitter in `RAPPIE/dataset`.

## An example to run lightGCN

run lightGCN on **twitter_glm** :

* change base directory

Change `ROOT_PATH` in `lightGCN/twitter_glm/code/world.py`

* command

` cd lightGCN/twitter_glm/code && python main.py --dataset='user_f' --view=r --layer=1 ` 

## An example to run RAPPIE

run RAPPIE on **twitter_glm** dataset:

Copy the results of `lightGCN/twitter_glm/result/user_f` to `RAPPIE/dataset/twitter_glm`.

Copy the role_features of `lightGCN/twitter_glm/data/user_f/role_features.pkl` to `RAPPIE/dataset/twitter_glm/role_features.pkl`.

* change base directory

if you want RAPPIE on **xxx_glm** dataset: Change `MODEL_PATH` in `RAPPIE/classifier_xxx_glm.py` to the path of `chatglm_6b`.

If you want to run RAPPIE on the **xxx_qwen** dataset: you need to obtain the corresponding data for this dataset and place it in **RAPPIE/dataset/xxxx_qwen**.

Due to the large size of the model files, please feel free to contact us separately if needed.

* command
` cd RAPPIE && python classifier_twitter_glm.py --dataset="RAPPIE/dataset/twitter_glm/" --batch_size=8 --learning_rate=7e-6 --epochs_num=15 --seed=3 --drouopt=0.5 `

## the experimental parameters on two datasets
* twitter_glm  
  dataset="RAPPIE/dataset/twitter_glm"

  dataset_type=Implicit 

  batch_size=8 

  learning_rate=7e-6 

  epochs_num=15 

  seed=3

  drouopt=0.5

* twitter_qwen  
  dataset="RAPPIE/dataset/twitter_qwen"

  batch_size=8 

  learning_rate=7e-6 

  epochs_num=15 

  seed=3

  drouopt=0.5 

* Weibo_glm  
  dataset="RAPPIE/dataset/weibo_glm"

  batch_size=8 

  learning_rate=2e-5 

  epochs_num=15 

  seed=6

* Weibo_qwen  
  dataset="RAPPIE/dataset/weibo_qwen"

  batch_size=8 

  learning_rate=2e-5 

  epochs_num=15 

  seed=6
