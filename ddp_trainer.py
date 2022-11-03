# encoding:utf-8
import os, sys, torch, time, datetime, numpy as np, cv2, argparse, random
from configs.train_cfg import Config

from utils.train_utils import create_logger, data_prefetcher, save_checkpoint, load_checkpoint, AverageMeter, Summary, matrix_accuracy, ProgressMeter, seconds_to_HMS
from models.backbone.cnn_utils import initialize_model, load_model
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from utils.visual import Visual#, show_board
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

def run_one_epoch(cfg, is_train, epoch_iter, model, data_loader, criterion, optimizer, device, logger, viser, rank):

    if is_train:
        model.train()
    else:
        model.eval()
    epoch_type = 'train' if is_train else 'test'

    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.3f', Summary.AVERAGE)

    data_len = len(data_loader)
    logger.info(f'{epoch_type} dataset len: {data_len * cfg.batch_size}, batch size: {cfg.batch_size // cfg.ngpu}')
    
    prefetcher = data_prefetcher(data_loader)
    data = prefetcher.next()
    epoch_loss = 0.0
    now_iter = 0
    # pred_right = 0
    # sample_num = 0
    start_time = time.time()

    bar = tqdm(total=data_len)
    progress = ProgressMeter(
                            len(data_loader),
                            [losses, top1],
                            prefix="Epoch: [{}]".format(epoch_iter)
                            )

    with torch.set_grad_enabled(is_train):
        while data is not None:
            now_iter += 1
            bar.update(1)
            img_tensor, label_tensor = data 
            img_tensor = img_tensor.to(device)
            label_tensor = label_tensor.to(device)

            optimizer.zero_grad()
            # print(img_tensor.shape)
            output = model(img_tensor)
            
            loss = criterion(output, label_tensor)

            now_loss = loss.item()
            epoch_loss += now_loss

            pred_output = torch.argmax(output, dim=1)
            gt_output = torch.argmax(label_tensor, dim=1)
            # now_right = torch.sum(pred_output==gt_output).item()
            now_right = sum([torch.equal(pred_output[i], gt_output[i]) for i in range(gt_output.shape[0])])

            acc1 = matrix_accuracy(output, label_tensor)
            losses.update(loss.item(), img_tensor.size(0))
            top1.update(acc1[0], img_tensor.size(0))

            if is_train:
                loss.backward()
                optimizer.step()

            # sample_num += len(label_tensor)
            # 可视化board
            # if now_iter % cfg.show_iter == 0 and rank == 0:
            #     show_img = show_board(img_tensor[0], pred_output[0], label_tensor[0])
            #     viser.img('goboard', show_img)

            now_time = time.time()
            now_cost_time = now_time - start_time
            predict_rest_left_time = (now_cost_time / now_iter) * (data_len - now_iter)
            if now_iter % cfg.print_iter == 0:
                
                logger.info(
                    f'{epoch_type} epoch {epoch_iter+1}, iter : {now_iter} / {data_len}, now_loss: {now_loss :.5f}, average loss : {losses.avg :.3f}, now acc: {now_right/pred_output.shape[0] :.5f}, average acc1:{top1.avg :.3f} %, 本轮已用时: {seconds_to_HMS(now_cost_time)}, 剩余时间预计: {seconds_to_HMS(predict_rest_left_time)}'
                    )
                
            data = prefetcher.next()
            
            if now_iter >= data_len:
                progress.display(now_iter + 1)
                logger.info(
                    f'{epoch_type} epoch {epoch_iter+1}, iter : {now_iter} / {data_len}, now_loss: {now_loss :.5f}, average loss : {losses.avg :.3f}, now acc: {now_right/pred_output.shape[0] :.5f}, average acc1:{top1.avg :.3f} %, 本轮已用时: {seconds_to_HMS(now_cost_time)}, 剩余时间预计: {seconds_to_HMS(predict_rest_left_time)}'
                    )
                # logger.info(
                #     f'{epoch_type} epoch {epoch_iter+1}, iter : {now_iter} / {data_len}, now_loss: {now_loss :.5f}, average loss : {epoch_loss/now_iter :.5f}, now acc: {now_right/pred_output.shape[0] :.5f}, average accuracy:{pred_right/sample_num :.5f}, 本轮已用时: {seconds_to_HMS(now_cost_time)}, 剩余时间预计: {seconds_to_HMS(predict_rest_left_time)}'
                #     )    
                
                
                bar.close()
                break
        
        top1.all_reduce()
        losses.all_reduce()
        progress.display(now_iter + 1)
    return losses.avg, top1.avg
    # return epoch_loss/data_len, pred_right/sample_num

def main_worker():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) train worker starting...")

    args = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {args}")

    nprocs = int(args["WORLD_SIZE"])
    print(f'init {local_rank} / {nprocs}!')
    cudnn.benchmark = True
    dist.init_process_group(backend="nccl")
    # 训练config加载
    cfg = Config()
    if cfg.dataloader_type == 'file':
        # 使用黑，白，空，1
        from dataloaders.ImageFileData import ImageFileDataset as MyDataset
    
    else:
        print(f"暂不支持 {cfg.in_feature_dim} dims 输入！")
        exit(1)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(local_rank)
    # log文件
    logger = create_logger(cfg.model_output_base_path, cfg.log_name, local_rank)
    print('now rank: ', local_rank)
    logger.info(f'now rank: {local_rank}')
    my_vis = Visual(cfg.model_output_base_path, log_to_file=cfg.vis_log)   
    # 初始化模型
    backbone = initialize_model(cfg.cnn_type, cfg.class_num)
    if cfg.resume_from_path:
        # resume 训练
        logger.info('resume from ' + cfg.resume_from_path)
        cfg.resume_epoch = load_checkpoint(backbone, None, cfg.resume_from_path)
    backbone.cuda(local_rank)        
    backbone = torch.nn.parallel.DistributedDataParallel(backbone, device_ids=[local_rank])

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    mini_batch_size = cfg.batch_size // nprocs
    # train_dataset = SGFDataset(cfg.train_file_path)
    train_dataset = MyDataset(cfg.train_file_path, data_transforms['train'])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_data_loader = DataLoader(train_dataset,batch_size=mini_batch_size, sampler=train_sampler, num_workers=cfg.workers, pin_memory=True)
    # test_dataset = SGFDataset(cfg.test_file_path)
    test_dataset = MyDataset(cfg.test_file_path, data_transforms['test'])
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_data_loader = DataLoader(test_dataset,batch_size=mini_batch_size, sampler=test_sampler, num_workers=cfg.workers, pin_memory=True)
    
    data_loaders = {
        'train': train_data_loader,
        'test': test_data_loader
    }
    
    # backbone = nn.DataParallel(backbone, device_ids=cfg.gpu_ids, output_device=cfg.gpu_ids[0])
    if local_rank == 0:
        # summary(backbone.module, torch.zeros((1, 3, 224, 224)).to(local_rank))
        logger.info(backbone)
        logger.info(f"Number of parameters: {sum([p.numel() for p in backbone.parameters() if p.requires_grad])}")


    if cfg.optims_type == 'adam':
        optimizer = torch.optim.Adam(backbone.parameters(), lr=cfg.lr)
    elif cfg.optims_type == 'adamW':
        optimizer = torch.optim.AdamW(backbone.parameters(), lr=cfg.lr)
    elif cfg.optims_type == 'Nadam':
        optimizer = torch.optim.NAdam(backbone.parameters(), lr=cfg.lr)
    elif cfg.optims_type == 'Radam':
        optimizer = torch.optim.RAdam(backbone.parameters(), lr=cfg.lr)
    elif cfg.optims_type == 'sgd':
        optimizer = torch.optim.SGD(backbone.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    elif cfg.optims_type == 'adadelta':
        optimizer = torch.optim.Adadelta(backbone.parameters(), lr=cfg.lr)
    else:
        print(f"optimizer type {cfg.optims_type} doesn't support! exit")
        exit(1)
    
    # criterion = nn.CrossEntropyLoss().cuda()
    # criterion = OnehotCrossEntropy().cuda()
    criterion = nn.BCELoss().cuda()

    logger.info(u'开始训练:')
    total_time = 0.0
    best_acc = 0.0
    train_best_acc = 0.0
    for train_iter in range(cfg.resume_epoch, cfg.epoch_num):
        logger.info(u'start %d / %d epoch : '%(train_iter+1, cfg.epoch_num))
        st_time = time.time()
        # 每个epoch打乱顺序
        train_sampler.set_epoch(train_iter)
        # 训练1 epoch
        train_loss, train_acc = run_one_epoch(cfg, True, train_iter, backbone, data_loaders['train'], criterion, optimizer, device, logger, my_vis, local_rank)
        learning_rate = optimizer.param_groups[0]["lr"]
        if local_rank == 0:
            my_vis.plot('train loss', train_loss)
            my_vis.plot('train acc', train_acc)
            my_vis.plot('learning rate', learning_rate)

            save_checkpoint(backbone, optimizer, train_iter, '%s/least_models.pth'%(cfg.model_output_base_path))
            if train_acc > train_best_acc:
                train_best_acc = train_acc
                save_checkpoint(backbone, optimizer, train_iter, '%s/train_best_models.pth'%(cfg.model_output_base_path))
        
        # 测试1 epoch
        test_loss, test_acc = run_one_epoch(cfg, False, train_iter, backbone, data_loaders['test'], criterion, optimizer, device, logger, my_vis, local_rank)
        if local_rank == 0:
            
            my_vis.plot('test loss', test_loss)
            my_vis.plot('test acc', test_acc)

        ed_time = time.time()
        now_time_use = ed_time - st_time
        total_time += now_time_use
        logger.info(u'epoch %d / %d finished. cost time: %s'%(train_iter+1, cfg.epoch_num, datetime.timedelta(seconds=now_time_use)))
        average_time_used = total_time / (train_iter+1)
        logger.info(u'预计还需要 %s'%(datetime.timedelta(seconds=average_time_used*(cfg.epoch_num-train_iter-1))))
        
        if local_rank == 0 and test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(backbone, optimizer, train_iter, '%s/best_models.pth'%(cfg.model_output_base_path))
            logger.info(f"best model: {train_iter} epoch, acc: {best_acc :.5f}")
    logger.info(u'总计用时 : %s'%(datetime.timedelta(seconds=total_time)))



if __name__ == "__main__":

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) train worker starting...")
    init_seeds(local_rank + 1)
    main_worker()
    # res = backbone(torch.zeros((1, 3, 224, 224)))
    # print(res.size(), res)
