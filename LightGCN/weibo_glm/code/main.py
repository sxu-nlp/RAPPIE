import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
import os
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

start_time = time.time()  # 记录开始时间
print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1


filepath = "../result/" + str(world.dataset) +'/'+ str(Recmodel.n_layers) + "layers"
filename = filepath + str(Recmodel.n_layers)+"_decay_"+str(world.config['decay'])+"_lr_"+str(world.config['lr'])+".txt"


# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    file_path = "../result/" + world.dataset + "/layer" + str(Recmodel.n_layers)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    if world.config['view'] == "r":
        embedding_file = file_path + "/user_embedding_repost_r.pkl"
    elif world.config['view'] == "c":
        embedding_file = file_path + "/user_embedding_comment_r.pkl"
    else:
        embedding_file = file_path + "/user_embedding_follow_r.pkl"

    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch %10 == 0:
            cprint("[TEST]")
            Procedure.Test(dataset, Recmodel, filename , epoch, w, world.config['multicore'])
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        torch.save(Recmodel.state_dict(), weight_file)
    Recmodel.saveEmbedding(embedding_file)
    print(Recmodel)
finally:
    if world.tensorboard:
        w.close()
        end_time = time.time()  # 记录结束时间
        print(f"Training ended at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"Total training time: {end_time - start_time:.2f} seconds")