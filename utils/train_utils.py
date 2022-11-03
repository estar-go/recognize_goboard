# encoding:utf-8
import os, numpy as np, random, cv2, logging, json, torch
from rich.logging import RichHandler

from rich import print 
from time import strftime, gmtime
import torch.distributed as dist

def seconds_to_HMS(seconds):
    return strftime("%H:%M:%S", gmtime(seconds))

class Summary(object):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':.5f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        total = torch.FloatTensor([self.sum, self.count]).cuda()
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def matrix_accuracy(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred_output = torch.argmax(output, dim=1)
        gt_output = torch.argmax(target, dim=1)
        now_right = torch.sum(torch.tensor([torch.equal(pred_output[i], gt_output[i]) for i in range(batch_size)])).mul(100.0 / batch_size)
        res = [now_right]
        return res

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        # self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_input = None
            return
        # with torch.cuda.stream(self.stream):
        #     self.next_data = self.next_data.cuda(non_blocking=True)
            
    def next(self):
        # torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data

def get_config_map(file_path):
    print(file_path)
    config_map = json.loads(open(file_path).read())
    
    config_map['batch_size'] *= len(config_map['gpu_ids'])
    return config_map

def create_logger(base_path, log_name, rank):

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    logger = logging.getLogger(f'{base_path}{log_name}')
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]
    
    if rank == 0:
        file_handler = logging.FileHandler(f'{base_path}{log_name}', 'w')
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    return logger

def get_show_result_img(gt_label, pred_label, conf=None):
    img = np.zeros((200, 800, 3), np.uint8)
    str_input = 'direct : gt: %d, pred : %d'%(gt_label * 90 , pred_label * 90)
    cv2.putText(img, str_input, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1., (255, 255, 255), 2)
    
    return img

def tensor_to_img(inp):
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)
    return inp[:,:,::-1]

def convert_show_cls_bar_data(acc_map, out_path, rename_map=None):
    mAP = 0.
    kl = acc_map.keys()
    name_l = [str(i) for i in kl]
    if rename_map:
        name_l = [str(rename_map[i]) for i in kl]
    acc_np = np.zeros((len(kl), 2), np.int32)

    with open(out_path, 'w') as f:

        for it, k in enumerate(kl):
            acc_np[it, :] = acc_map[k]
            t_str ='now cls id: %5s, total : %5d, right: %5d, wrong: %5d, Acc %.3f'%(name_l[it], acc_map[k][0] + acc_map[k][1], acc_map[k][0], acc_map[k][1], acc_map[k][0]/(acc_map[k][0] + acc_map[k][1]))
            print(t_str)
            f.write(t_str + '\n')

            mAP += acc_map[k][0]/(acc_map[k][0] + acc_map[k][1])
        if len(kl) != 0:
            mAP /= len(kl)
            print('*'*20, 'mAP is : %.5f'%(mAP), '*'*20)
            f.write('*'*20 + 'mAP is : %.5f'%(mAP) + '*'*20)
        else:
            print('kl len is 0!!!')

        leg_l = ['right', 'wrong']
    return acc_np, leg_l, name_l

def save_checkpoint(model, optimizer, epoch, save_path):
    state = {'net':model.module.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
    torch.save(state, save_path)


def load_checkpoint(model, optimizer, load_path):
    checkpoint = torch.load(load_path, map_location='cpu')
    model.load_state_dict(checkpoint['net'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1
    # return model, optimizer, start_epoch
    return start_epoch

def tensor2np(tensor):
    if tensor.is_cuda:
        return tensor.cpu().numpy()
    return tensor.numpy()

