import os
import random
from typing import OrderedDict
import numpy as np
import logging
import argparse

import cv2
from sklearn import metrics
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from os.path import join

from util import config
from tqdm import tqdm
from models.p2p import P2P
from models.layers.utils import fps
from util.rotate import rotate_point_clouds, rotate_theta_phi
from sklearn.metrics import confusion_matrix

if not hasattr(torch, 'pi'):
    import math
    torch.pi = math.pi


cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def worker_init_fn(worker_id):
    random.seed(1463 + worker_id)
    np.random.seed(1463 + worker_id)
    torch.manual_seed(1463 + worker_id)


def get_parser():
    parser = argparse.ArgumentParser(description='BPNet')
    parser.add_argument('--config', type=str, default='config/scannet/bpnet_5cm.yaml', help='config file')
    parser.add_argument('opts', help='see config/scannet/bpnet_5cm.yaml for all options', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def main_process():
    return not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    cudnn.benchmark = True

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)

    print(
        'torch.__version__:%s\ntorch.version.cuda:%s\ntorch.backends.cudnn.version:%s\ntorch.backends.cudnn.enabled:%s' % (
            torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.backends.cudnn.enabled))

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.test_gpu)
    if len(args.test_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
        args.use_apex = False

    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.test_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                                rank=args.rank)

    model = P2P(args, is_test=True)
    if main_process():
        global logger
        logger = get_logger()
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info(model)

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.test_batch_size = int(args.test_batch_size / ngpus_per_node)
        args.test_workers = int(args.test_workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu])
    else:
        model = model.cuda()

    # ####################### Data Loader ####################### #
    if args.data_name == 'modelnet':
        from dataset.modelnet import ModelNet
        val_data = ModelNet(config=args, split='test')
        val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_batch_size,
                                                 shuffle=False, num_workers=args.workers, pin_memory=True,
                                                 drop_last=False, sampler=val_sampler)

    elif args.data_name == 'scanobjectnn' or args.data_name == 'scanobjectnn_hardest':
        from dataset.scanobjectnn import ScanObjectNN
        val_data = ScanObjectNN(config=args, subset='test')
        val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_batch_size,
                                                    shuffle=False, num_workers=args.workers, pin_memory=True,
                                                    drop_last=False, sampler=val_sampler)

    elif args.data_name == 'shapenetpart':
        
        from dataset.shapenetpart import shapenetpart
        root = './data'
        dataset_name = 'shapenetpart'
        split = 'test'
        
        val_data = shapenetpart(root=root, dataset_name=dataset_name, num_points=2048, split=split)
        val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_batch_size,
                                                    shuffle=False, num_workers=args.test_workers, pin_memory=True,
                                                    drop_last=False, sampler=val_sampler)

    elif args.data_name == 'shapenetcore':
        from dataset.shapenetcore import ShapeNetCore
        val_data = ShapeNetCore(config=args, split='test')
        val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_batch_size,
                                                    shuffle=False, num_workers=args.test_workers, pin_memory=True,
                                                    drop_last=False, sampler=val_sampler)

    else:
        raise Exception('Dataset not supported yet'.format(args.data_name))


    if os.path.isfile(args.model_path):
        if main_process():
            logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage.cuda())
        
        state_dict = OrderedDict({key.replace("module.", ""): value for key, value in checkpoint['state_dict'].items()})
    
        model.load_state_dict(state_dict, strict=True)
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))

    # ####################### Test ####################### #
    if args.data_name == 'modelnet' or args.data_name == 'scanobjectnn' or args.data_name == 'scanobjectnn_hardest' or args.data_name == 'shapenetpart' or args.data_name == 'shapenetcore' :
        test_cls(model, val_loader)
    else:
        raise Exception('Dataset not supported yet'.format(args.data_name))


def test_cls(model, val_data_loader):
    torch.backends.cudnn.enabled = False  # for cudnn bug at https://github.com/pytorch/pytorch/issues/4107

    # here we produce the point cloud after rotating into 80 angles
    theta = np.linspace(0, 2, 11)
    phi = np.linspace(-0.4, -0.2, 4)
    v_theta, v_phi = np.meshgrid(theta[:10], phi)
    angles = np.stack([v_theta, v_phi], axis=-1).reshape(-1, 2)
    angles = torch.from_numpy(angles) * torch.pi
    rotation_matrixs = rotate_theta_phi(angles)

    with torch.no_grad():
        model.eval()
        input_points = []
        fps_points = []
        labels = []
        preds = []
        confs = []
        if main_process():
            pbar = tqdm(total=len(val_data_loader))
        for i, batch_data in enumerate(val_data_loader):
            if main_process():
                pbar.update(1)
            points = batch_data[0].cuda()

            input_points = points

            points = fps(points, args.npoints)

            fps_points = points

            outputs = []
            confidences2 = []
            predictions = []
            # vote
            for rotation_matrix in rotation_matrixs:
                input_pc = rotate_point_clouds(points, rotation_matrix, use_normals=args.use_normals)
                output = model(input_pc, original_pc = points)
                confidence2 = np.max(torch.softmax(output, dim=-1).detach().cpu().numpy(), axis=1)
                prediction = torch.argmax(output, 1).detach().cpu().numpy()
                outputs.append(output.detach().unsqueeze(0))
                confidences2.append(confidence2)
                predictions.append(prediction)
            
            outputs = torch.cat(outputs, dim=0).mean(0)
            confidences = torch.softmax(outputs, dim=-1)
            confidence = np.max(confidences.detach().cpu().numpy(), axis=1)
            confidences2 = np.concatenate(confidences2).reshape(40,args.test_batch_size).swapaxes(0,1)
            predictions = np.concatenate(predictions).reshape(40,args.test_batch_size).swapaxes(0,1)
            preds.append(torch.argmax(outputs, 1).detach().cpu().numpy())
            labels.append(batch_data[1].numpy())
            confs.append(np.expand_dims(confidence, axis=1))
            torch.cuda.empty_cache()
            break

        if main_process():
            pbar.close()
        labels = np.concatenate(labels).flatten()
        preds = np.concatenate(preds)
        confs = np.concatenate(confs).flatten()
        #conf_mtx = confusion_matrix(labels,preds,normalize='true')
        np.save(join(args.save_folder, 'gt.npy'), labels)
        np.save(join(args.save_folder, 'pred.npy'), preds)
        np.save(join(args.save_folder, 'conf.npy'), confs)
        np.save(join(args.save_folder, 'all_conf.npy'), confidences2)
        np.save(join(args.save_folder, 'all_pred.npy'), predictions)
       # np.save(join(args.save_folder, 'conf_mtx.npy'), conf_mtx)
        np.save(join(args.save_folder, 'input_points.npy'), input_points.detach().cpu().numpy())
        np.save(join(args.save_folder, 'fps_points.npy'), fps_points.detach().cpu().numpy())
        oAcc = metrics.accuracy_score(labels, preds) * 100
        mAcc = metrics.balanced_accuracy_score(labels, preds) * 100
        
        if main_process():
            print("Test overall accuracy: ", oAcc)
            print("Test mean accuracy: ", mAcc)
            

if __name__ == '__main__':
    main()
