import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms, models
import os
import time
import math
import scipy
import einops
import random
import argparse
import warnings
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
from matplotlib import pyplot as plt
from loss import MaskedSmoothL1Loss, MaskedSmoothL1LossEqual
from datasets import CardiacSeqDataset
from models.seq_model import IJEPA_Ultrastar_Model
from utils.lr_decay import param_groups_lrd, add_weight_decay

parser = argparse.ArgumentParser()
parser.add_argument('--base_model', type=str, default='deit')
parser.add_argument('--pretrain_weights', type=str, default=None, help='initial weights path')
parser.add_argument('--encoderpath', type=str, default=None, help='initial weights path')
parser.add_argument('--lr', type=float, default=5e-5, help='learn rate')
parser.add_argument('--lr_f', type=float, default=1e-6, help='final learn rate')
parser.add_argument('--lr_sched', type=str, default='cos', help='cosine annealing learning rate')
parser.add_argument('--data_root', type=str, default='/nfs/data4/jhj', help='data path, in this path, we can just see train and val folder in it')
parser.add_argument('--image_folder', type=str, default='modified_dataset')
parser.add_argument('--timestep', type=int, default=2)
parser.add_argument('--logs', type=str, default='/nfs/data4/jhj/nav_logs', help='model save path')
parser.add_argument('--save_period', type=int, default=1, help='Number of epochs to save the model')
parser.add_argument('--num_class', type=int, default=6, help='number of class to classify')
parser.add_argument('--epochs', type=int, default=5, help='total epoch to train')
parser.add_argument('--warmup_epochs', type=int, default=0, help='total epoch to train')
parser.add_argument('--un_freeze_epoch', type=int, default=10000, help='after this epoch, unfreeze all the paramiters')
parser.add_argument('--batch-size', type=int, default=4096, help='total batch size')
parser.add_argument("--local_rank", default=-1, type=int, help="Don't change it")
parser.add_argument("--distributed", default=True, type=bool, help="Use DDP for training")
parser.add_argument("--amp", action='store_true', help="Mixed precision training, only used in DDP")
parser.add_argument("--sync_bn", default=False, type=bool, help='Use sync batch normalization, only used in DDP')
parser.add_argument('--num-workers', type=int, default=4, help='number workers')
parser.add_argument('--prefetch_factor', type=int, default=2, help='number workers')
parser.add_argument("--device_id", default='0, 1, 2, 3, 4, 5', type=str,
                help="Numbers of cuda want to use. if only have one GPU, default=0")
parser.add_argument('--img-size', nargs='+', type=tuple, default=(224, 224), help='[train, test] image sizes')
parser.add_argument('--view_number', type=int, default=10, help='learn rate')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--evaluate', action='store_true', help='evaluate the model on validation set')
parser.add_argument('--freeze_encoder', action='store_true', help='freeze the parameter inside encoder')
parser.add_argument('--sample_mode', type=str, default='segmental_random')
parser.add_argument("--equal_loss", action='store_true', help="whether to average the loss with mask")
parser.add_argument('--drop_path_rate', type=float, default=0.)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--layer_decay', type=float, default=0.75)
parser.add_argument('--proj_type', type=str, default='mlp')
parser.add_argument('--topk', type=int, default=128)
args = parser.parse_args()

def main():
    args = parser.parse_args()

    exp_name = f'topk{args.topk}_ts{args.timestep}_e{args.epochs}_lr{args.lr}_lrf{args.lr_f}_bs{args.batch_size}'
    args.logs = os.path.join(args.logs, exp_name)

    if os.path.exists(args.logs):
        num_exp = len(os.listdir(args.logs))
        args.logs = os.path.join(args.logs, f'no{num_exp + 1}')
    else:
        args.logs = os.path.join(args.logs, 'no1')
    os.makedirs(args.logs)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
        if ngpus_per_node == 1 and args.dist_backend == "nccl":
            warnings.warn("nccl backend >=2.5 requires GPU count>1, see https://github.com/NVIDIA/nccl/issues/103 perhaps use 'gloo'")
    else:
        ngpus_per_node = 1

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    local_rank = args.gpu
    print(local_rank)

    if local_rank == 0:
        print(args)
        loss_dir = os.path.join(args.logs, 'loss')
        if not os.path.exists(loss_dir):
            os.makedirs(loss_dir)
        loss_history = LossHistory(loss_dir)
    else:
        loss_history = None

    train_transform = transforms.Compose(
        [transforms.Resize(args.img_size),
         transforms.RandomApply([transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.8, 1.2))], p=0.6),
         transforms.RandomApply([transforms.GaussianBlur(kernel_size=(random.choice([5, 7, 9, 11, 13])), sigma=random.uniform(0.1, 2))], p=0.6),
         transforms.ToTensor(),
         transforms.Normalize([0.193, 0.193, 0.193], [0.224, 0.224, 0.224])
         ])

    test_transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.193, 0.193, 0.193], [0.224, 0.224, 0.224])
    ])


    train_dataset = CardiacSeqDataset(
        root=args.data_root,
        image_folder=args.image_folder,
        train=True,
        transform=train_transform,
        timestep=args.timestep,
        sample_mode=args.sample_mode,
        topk=args.topk
    )
    test_dataset = CardiacSeqDataset(
        root=args.data_root,
        image_folder=args.image_folder,
        train=False,
        transform=test_transform,
        timestep=args.timestep,
        sample_mode=args.sample_mode,
        topk=args.topk
    )
        
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        batch_size = args.batch_size // ngpus_per_node
        shuffle = False
    else:
        batch_size = args.batch_size
        train_sampler, val_sampler = None, None
        shuffle = True

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle,
                                               num_workers=args.num_workers,
                                               pin_memory=True,
                                               drop_last=True,
                                               sampler=train_sampler,
                                               persistent_workers=True,
                                               prefetch_factor=args.prefetch_factor)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                              drop_last=False,
                                              sampler=val_sampler,
                                              persistent_workers=True,)

    if args.base_model == 'ijepa':
        import models.vision_transformer as vits
        base_model = vits.__dict__['vit_small'](img_size=[224],
                patch_size=16, drop_path_rate=args.drop_path_rate)
        ckpt = torch.load(args.encoderpath, map_location='cpu', weights_only=True)
        if 'target_encoder' in ckpt:
            state_dict = ckpt["target_encoder"]
        elif 'model' in ckpt:
            state_dict = ckpt['model']
            state_dict = {k.replace('target_encoder.', ''): v for k, v in state_dict.items() if k.startswith('target_encoder.')}
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_key = k[7:] 
            else:
                new_key = k
            new_state_dict[new_key] = v
        msg = base_model.load_state_dict(new_state_dict, strict=True)
        print(f'Load Ijepa ckpt with missing {msg.missing_keys}')
    else:
        raise NotImplementedError
 
    net = IJEPA_Ultrastar_Model(base_model, proj_type=args.proj_type)
       
    if args.pretrain_weights:
        pretrain_weights = torch.load(args.pretrain_weights, map_location='cpu')
        net.load_state_dict(pretrain_weights)
        print('Pretrain weights loaded from {}'.format(args.pretrain_weights))
        
    if args.freeze_encoder:
        print('freeze_encoder')
        for p in net.feature_model.parameters():
            p.requires_grad = False

    if local_rank == 0:
        total_params = sum(p.numel() for p in net.parameters()) / 1e6
        print('Total number of [Net] parameters: {} M'.format(total_params))

    if args.sync_bn and ngpus_per_node > 1 and args.distributed:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        if local_rank == 0:
            print('use sync_bn')
    elif args.sync_bn:
        if local_rank == 0:
            print("Sync_bn is not support in one gpu or not distributed.")

    if args.distributed:  
        torch.cuda.set_device(local_rank)
        net = net.cuda(local_rank)
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        net_without_ddp = net.module
        cudnn.benchmark = True
    else:
        if torch.cuda.device_count() > 1:  
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True
            net = net.cuda()
            net_without_ddp = net.module
        else:  
            cudnn.benchmark = True
            net = net.cuda()
            net_without_ddp = net

    if args.amp:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    encoder_param_groups = param_groups_lrd(net_without_ddp.feature_model, args.weight_decay,
                                            no_weight_decay_list=['pos_embed', 'cls_token', 'head.cls_query',
                                                                    'prompt'], layer_decay=args.layer_decay)
    head_param_groups = add_weight_decay(net_without_ddp.fc_out, weight_decay=args.weight_decay)
    geo_content_proj_param_groups = add_weight_decay(net_without_ddp.refine_blocks, weight_decay=args.weight_decay)
    geo_bias_proj_param_groups = add_weight_decay(net_without_ddp.locator_attn, weight_decay=args.weight_decay)
    action_encoder_param_groups = add_weight_decay(net_without_ddp.action_encoder, weight_decay=args.weight_decay)
    param_groups = encoder_param_groups + head_param_groups + geo_bias_proj_param_groups + geo_content_proj_param_groups + action_encoder_param_groups

    optimizer = torch.optim.AdamW(params=param_groups, lr=args.lr, betas=(0.9, 0.999), eps=1e-6)

    if args.equal_loss:
        criterion = MaskedSmoothL1LossEqual()
    else:
        criterion = MaskedSmoothL1Loss()

    if args.evaluate:
        val(net, test_loader, criterion, 0, args, local_rank)

    best_epoch = 0
    best_mean_mae = 0
    best_mae = 0
    mae_loss = 1000
    for epoch in range(1, args.epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_loss = train(net, net_without_ddp, train_loader, optimizer, criterion, epoch, args, scaler, local_rank)
        val_loss, val_mae = val(net, test_loader, criterion, epoch, args, local_rank)
        mae_mean = val_mae.mean()

        if mae_mean < mae_loss:
            mae_loss = mae_mean
            best_epoch = epoch
            best_mean_mae = mae_mean
            best_mae = val_mae
            if local_rank == 0:
                best_epoch = epoch
                print('save MAE best model to logs!')
                torch.save(net_without_ddp.state_dict(), os.path.join(args.logs, 'best_mae.pth'))

        if local_rank == 0:
            loss_history.append_loss(train_loss, val_loss)
            print('current best epoch:', best_epoch)
            print('lr:', optimizer.state_dict()['param_groups'][0]['lr'])

    if local_rank == 0:
        with open(os.path.join(args.logs, 'result.txt'), 'a+') as f:
            f.writelines('\nBest epoch: %f, Best mean mae: %f \n' % (best_epoch, best_mean_mae))
            for list in best_mae.cpu().numpy().tolist():
                for value in list:
                    f.writelines('%f,' % value)
                f.writelines('\n')


def train(net, net_without_ddp, train_loader, optimizer, criterion, epoch, args, scaler, local_rank):
    if local_rank == 0:
        print("Start train, Epoch: %d" % epoch)
    net.train()
    total_loss = 0
    if epoch > args.un_freeze_epoch:
        for param in net.parameters():
            param.requires_grad = True
        if local_rank == 0:
            print('unfreeze backbone')

    for batch_idx, (imgs, acts, label, mask) in enumerate(tqdm(train_loader)):
        lr = adjust_learning_rate(optimizer, epoch + batch_idx / len(train_loader), args)
        imgs = imgs.cuda(local_rank)
        acts = acts.cuda(local_rank)
        label = label.cuda(local_rank)
        mask = mask.cuda(local_rank)
        optimizer.zero_grad()

        if not args.amp:
            out = net(imgs, acts)  
            loss = criterion(out, label, mask)
            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast(enabled=scaler is not None):
                out = net(imgs, acts) 
                mse_loss = criterion(out, label, mask)
                loss = mse_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += loss.item()
        if local_rank == 0:
            while True:
                try:
                    with open(os.path.join(args.logs, 'result.txt'), 'a+') as f:
                        f.writelines("Epoch:%d [%d|%d] lr:%f, loss:%f \n" % (
                            epoch, batch_idx + 1, len(train_loader), lr, loss.mean()))
                except OSError as e:
                    import time
                    time.sleep(1)
                else:
                    break

    epoch_loss = total_loss / len(train_loader)
    
    if epoch % args.save_period == 0 and local_rank == 0:
        print('save model to logs')
        torch.save(net_without_ddp.state_dict(),
                   os.path.join(args.logs, 'epoch_%d_loss_%f.pth') % (epoch, total_loss))  

    if local_rank == 0:
        while True:
            try:
                with open(os.path.join(args.logs, 'result.txt'), 'a+') as f:
                    f.writelines('\nEpoch: %d, total loss: %f, epoch loss: %f' % (epoch, total_loss, epoch_loss))
            except OSError as e:
                import time
                time.sleep(1)
            else:
                break
        print('\nEpoch: %d, total loss: %f, epoch loss: %f' % (epoch, total_loss, epoch_loss))
        
    return epoch_loss


def val(net, test_loader, criterion, epoch, args, local_rank):
    if local_rank == 0:
        print("Start valid, Epoch: %d" % epoch)
    net.eval()
    total_loss = 0
    ae = torch.tensor([0.]*args.view_number*6).cuda(local_rank) 
    ae_count = torch.zeros(60, dtype=torch.long).cuda(local_rank)
    with torch.no_grad():

        for batch_idx, (imgs, acts, label, mask) in enumerate(tqdm(test_loader)):
            imgs = imgs.cuda(local_rank)
            acts = acts.cuda(local_rank)
            label = label.cuda(local_rank)
            mask = mask.cuda(local_rank)
            out = net(imgs, acts)  
            
            masked_label = label*mask
            masked_out = out*mask
            ae_count = ae_count + mask.sum(0)
            ae += abs(masked_label.float() - masked_out).sum(axis=0)  
            mse_loss = criterion(out, label, mask)
            loss = mse_loss
            total_loss += loss.item()

        dist.barrier()
        dist.all_reduce(ae, dist.ReduceOp.SUM, async_op=False)
        dist.all_reduce(ae_count, dist.ReduceOp.SUM, async_op=False)
        
        mae = (ae / ae_count).view(10, 6)
        epoch_loss = total_loss / len(test_loader)
        
    if local_rank == 0:
        while True:
            try:
                with open(os.path.join(args.logs, 'result.txt'), 'a+') as f:
                    f.writelines('\nVal total loss: %f, epoch loss: %f, mean mae: %f \n' % (total_loss, epoch_loss, mae.mean()))
                    for list in mae.cpu().numpy().tolist():
                        for value in list:
                            f.writelines('%f,' % value)
                        f.writelines('\n')
            except OSError as e:
                import time
                time.sleep(1)
            else:
                break
        print('Val epoch loss: %f, mean mae: %f' % (epoch_loss, mae.mean()))
        print(f'Val MAE: {mae}')

    return epoch_loss, mae


class LossHistory():
    def __init__(self, log_dir, val_loss_flag=True, input_shape=(224, 224)):
        self.log_dir = log_dir
        self.val_loss_flag = val_loss_flag

        self.losses = []
        if self.val_loss_flag:
            self.val_loss = []

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def append_loss(self, loss, val_loss=None):
        self.losses.append(loss)
        if self.val_loss_flag:
            self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "train_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        if self.val_loss_flag:
            with open(os.path.join(self.log_dir, "val_loss.txt"), 'a') as f:
                f.write(str(val_loss))
                f.write("\n")

        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        if self.val_loss_flag:
            plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')
            if self.val_loss_flag:
                plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--',
                         linewidth=2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))
        plt.cla()
        plt.close("all")


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if args.lr_sched == 'cos':
        if epoch < args.warmup_epochs:
            lr = args.lr * epoch / args.warmup_epochs
        else:
            lr = args.lr_f + (args.lr - args.lr_f) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    elif args.lr_sched == 'const':
        if epoch < args.warmup_epochs:
            lr = args.lr * epoch / args.warmup_epochs
        else:
            lr = args.lr
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

if __name__ == '__main__':
    main()



