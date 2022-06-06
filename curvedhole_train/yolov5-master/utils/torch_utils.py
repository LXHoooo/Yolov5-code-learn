# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
PyTorch utils
"""

import math
import os
import platform   # 提供获取操作系统相关信息的模块
import subprocess  # 子进程定义及操作的模块
import time
import warnings
from contextlib import contextmanager  # 用于进行上下文管理的模块
from copy import deepcopy
from pathlib import Path  # path将str转换为path对象，使字符路径易于操作的模块

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from utils.general import LOGGER, file_update_date, git_describe

try:
    import thop  # for FLOPs computation 用于pytorch模型的FLOPS计算工具模块
except ImportError:
    thop = None

# Suppress PyTorch warnings
warnings.filterwarnings('ignore', message='User provided device_type of \'cuda\', but CUDA is not available. Disabling')


# 这个函数是用来处理模型进行分布式训练时的同步问题
@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    用于train.py中
    用于处理模型进行分布式训练时同步问题
    基于torch.distributed.barrier()函数的上下文管理器，为了完成数据的正常同步操作
    :param local_rank:代表当前进程号，0代表主进程，1、2、3代表子进程
    :return:
    """
    # Decorator to make all processes in distributed training wait for each local_master to do something
    if local_rank not in [-1, 0]:
        # 如果执行create_dataloader()函数的进程不hi主进程，即rank不等于0或者-1
        # 上下文管理器会执行相应的torch.distributed.barrier()，这是一个阻塞栏，
        # 让此进程处于等待状态，等待所有进程到达栅栏处（包括主进程数据处理完毕）
        dist.barrier(device_ids=[local_rank])
    yield  # yield语句中，中断后执行上下文代码，然后返回到此处继续往下执行
    if local_rank == 0:
        # 如果执行create_dataloader()函数的进程时主进程，其会直接去读取数据并处理，
        # 然后其处理结束之后会接着遇到torch.distributed.barrier(),
        # 此时，所有进程都到达了当前的栅栏处，这样所有进程就达到了同步，并同时得到释放
        dist.barrier(device_ids=[0])


def device_count():
    # Returns number of CUDA devices available. Safe version of torch.cuda.device_count(). Only works on Linux.
    assert platform.system() == 'Linux', 'device_count() function only works on Linux'
    try:
        cmd = 'nvidia-smi -L | wc -l'
        return int(subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout.decode().split()[-1])
    except Exception:
        return 0


# 自动选择系统设备的操作
def select_device(device='', batch_size=0, newline=True):
    # device = 'cpu' or '0' or '0,1,2,3'
    """
    广泛用于train.py、test.py、detect.py等文件中
    用于选择模型训练的设备，并输出日志信息
    :param device: 输入的设备，device = 'cpu' or '0' or '0,1,2,3'
    :param batch_size: 一个批次的图片个数
    :param newline:
    :return:
    """
    s = f'YOLOv5 🚀 {git_describe() or file_update_date()} torch {torch.__version__} '  # string
    device = str(device).strip().lower().replace('cuda:', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    if cpu:
        # 如果cpu=True，就强制使用cpu，令torch.cuda.is_available() = False
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        # 如果输入device不为空，device=GPU,直接设置cuda环境
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
        # 检查cuda的可用性，如果不可用则终止程序
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    # 输入device为空，自行根据计算机情况选择相应设备，先看GPU，没有就cpu
    # 如果cuda可用且输入device != cpu，则cuda=True,反正GPU，没有就cpu
    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count，所有可用的GPU设备数量
        # 检查是否有GPU设备，且batch_size是否可以能被显卡数目整除
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            # 如果不能则关闭程序
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(s) + 1)
        # 满足所有条件，s加上所有显卡的信息
        for i, d in enumerate(devices):
            # p:每个可用显卡的相关属性
            p = torch.cuda.get_device_properties(i)
            # 显示信息s加上每张显卡的属性信息
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
    else:
        # cuda不可用，显示信息就加上cpu
        s += 'CPU\n'

    if not newline:
        s = s.rstrip()
    # 将显示信息s加入logger日志文件中
    LOGGER.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji-safe
    # 如果cuda可用，就返回第一张显卡的名称
    return torch.device('cuda:0' if cuda else 'cpu')


# 用于在进行分布式操作时，精确的获取当前时间
def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


# 没用到，用来输出某个网络结构的一下信息
def profile(input, ops, n=10, device=None):
    # YOLOv5 speed/memory/FLOPs profiler
    #
    # Usage:
    #     input = torch.randn(16, 3, 640, 640)
    #     m1 = lambda x: x * torch.sigmoid(x)
    #     m2 = nn.SiLU()
    #     profile(input, [m1, m2], n=100)  # profile over 100 iterations

    results = []
    device = device or select_device()
    print(f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}"
          f"{'input':>24s}{'output':>24s}")

    for x in input if isinstance(input, list) else [input]:
        x = x.to(device)
        x.requires_grad = True
        for m in ops if isinstance(ops, list) else [ops]:
            m = m.to(device) if hasattr(m, 'to') else m  # device
            m = m.half() if hasattr(m, 'half') and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m
            tf, tb, t = 0, 0, [0, 0, 0]  # dt forward, backward
            try:
                flops = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # GFLOPs
            except Exception:
                flops = 0

            try:
                for _ in range(n):
                    t[0] = time_sync()
                    y = m(x)
                    t[1] = time_sync()
                    try:
                        _ = (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum().backward()
                        t[2] = time_sync()
                    except Exception:  # no backward method
                        # print(e)  # for debug
                        t[2] = float('nan')
                    tf += (t[1] - t[0]) * 1000 / n  # ms per op forward
                    tb += (t[2] - t[1]) * 1000 / n  # ms per op backward
                mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0  # (GB)
                s_in = tuple(x.shape) if isinstance(x, torch.Tensor) else 'list'
                s_out = tuple(y.shape) if isinstance(y, torch.Tensor) else 'list'
                p = sum(list(x.numel() for x in m.parameters())) if isinstance(m, nn.Module) else 0  # parameters
                print(f'{p:12}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}{str(s_in):>24s}{str(s_out):>24s}')
                results.append([p, flops, mem, tf, tb, s_in, s_out])
            except Exception as e:
                print(e)
                results.append(None)
            torch.cuda.empty_cache()
    return results


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def initialize_weights(model):
    """
    在yolo.py的model类中的init函数被调用
    用于初始化模型权重
    :param model:
    :return:
    """
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:  # 如果是二维卷积就跳过，或者使用何凯明初始化
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:  # 如果是bn层，就设置相关参数:eps和momentum
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            # 如果是这几类激活函数，inplace插值就赋为True
            # inplace = True指进行原地操作，对于上层网络传递下来的tensor直接进行修改，不需要另外赋值变量
            # 这样可以节省运算内存，不用多储存变量
            m.inplace = True


# 没用到
def find_modules(model, mclass=nn.Conv2d):
    # Finds layer indices matching module class 'mclass'
    return [i for i, m in enumerate(model.module_list) if isinstance(m, mclass)]


# 用来计算模型的稀疏成都sparsity，返回模型整体的稀疏性，会被prune剪枝函数中被调用
def sparsity(model):
    # Return global model sparsity
    a, b = 0, 0
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a


# 用于模型剪枝的，代码中并没有用到
def prune(model, amount=0.3):
    # Prune model to requested global sparsity
    """
    可用于test.py和detect.py中进行模型剪枝
    :param model:
    :param amount:
    :return:
    """
    import torch.nn.utils.prune as prune
    print('Pruning model... ', end='')
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name='weight', amount=amount)  # prune
            prune.remove(m, 'weight')  # make permanent
    print(' %.3g global sparsity' % sparsity(model))


def fuse_conv_and_bn(conv, bn):
    """
    在yolo.py中Model类的fuse函数中调用
    融合卷积层和BN层（测试推理使用）
    方法：卷积层还是正常定义，但是卷积层的参数w,b要改变，通过只改变卷积参数，达到conv+bn的效果
    w = w_bn * w_conv b = w_bn * b_conv + b_bn(可以证明)
    :param conv:torch支持的卷积层
    :param bn:torch支持的bn层
    :return:
    """
    # Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # w_conv：卷积层的w参数
    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def model_info(model, verbose=False, img_size=640):
    """
    用于yolo.py文件的model类的ynfo函数
    输出模型的所有信息，包括：所有层数量，模型总参数量，需要求梯度的总参数量，img_size大小的model的浮点计算量GFLOPs
    :param model:模型
    :param verbose:是否输出每一层的参数parameters的相关信息
    :param img_size: int or list img_size=640 or img_size=[640,320]
    :return:
    """
    # n_p:模型model的总参数 number parameters
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    # n_g:模型model的参数中需要求梯度的参数量
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        # 表头：‘layer’,'name','gradient'   , 'parameters' ,'shape'        ,'mu'        ,'sigma'
        #      '第几层'，'层名','是否需要求梯度'，'当前层参数数量'，‘当前层参数shape’‘当前参数均值’,'当前层参数方差'
        print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        # 按表头输出每一层的参数parameters的相关信息
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPs
        from thop import profile  # 导入计算浮点计算量FLOPs的工具包
        # stride，模型的最大下采样率，有[8,16,32]，所以stride=32
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        # 模拟一张输入图片，shape=(1,3,32,32)，全是0
        img = torch.zeros((1, model.yaml.get('ch', 3), stride, stride), device=next(model.parameters()).device)  # input
        # 调用profile计算输入图片img=(1,3,32,32)时，但钱模型的浮点计算量GFLOPs
        # profile求出来的浮点计算量时FLOPs/1E9 => GFLOPs, *2是因为profile函数默认求的就是模型为float64时的浮点计算量
        # 而我们传入的模型一般都是float32，所以乘以2
        flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2  # stride GFLOPs
        # expand img_size -> [img_size,img_size]=[640,640]
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # expand if int/float
        # 根据img=(1,3,32,32)的浮点计算量flops推算出640x640的图片的浮点计算量
        fs = ', %.1f GFLOPs' % (flops * img_size[0] / stride * img_size[1] / stride)  # 640x640 GFLOPs
    except (ImportError, Exception):
        fs = ''

    name = Path(model.yaml_file).stem.replace('yolov5', 'YOLOv5') if hasattr(model, 'yaml_file') else 'Model'
    LOGGER.info(f"{name} summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")


def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    """
    用于yolo.py文件中model类的forward_augment函数中
    实现对图片的缩放操作
    :param img:原图
    :param ratio:缩放比例 默认=1.0原图
    :param same_shape:缩放之后尺寸是否要求的大小（必须是gs=32的倍数）
    :param gs:最大的下采样率 32，所以缩放后的图片的shape必须是gs=32的倍数
    :return:
    """
    # Scales img(bs,3,y,x) by ratio constrained to gs-multiple
    if ratio == 1.0:  # 如果缩放比例为1.0,直接返回原图
        return img
    else:
        # h,w：原图的高和宽
        h, w = img.shape[2:]
        # s:缩放后图片的新尺寸
        s = (int(h * ratio), int(w * ratio))  # new size
        # 直接使用torch自带的F.interpolate(上采样下采样函数)插值函数进行resize
        # F.interpolate:可以给定size或者scale_factor来进行上下采样
        #                mode = 'bilinear':双线性插值 nearest:最近邻
        #                align_corner:是否对齐input和output的角点像素
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
        if not same_shape:  # pad/crop img
            # 缩放之后要是尺寸和要求的大小（必须是gs=32的倍数）不同，再对其不相交的部分进行pad
            # 而pad的值就是imagenet的mean
            # math.ceil():向上取整，这里除以gs向上取整再乘以gs是为了保证h、w都是gs的倍数
            h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


def copy_attr(a, b, include=(), exclude=()):
    """
    在modelEMA函数和yolo.py中model类的autoshape函数中调用
    复制b的属性（这个属性必须在include中而不在exclude中）
    :param a:对象a（待赋值）
    :param b:对象b（赋值）
    :param include:可以赋值的属性
    :param exclude:不能赋值的属性
    :return:
    """
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            # 将对象b的属性k赋值给a
            setattr(a, k, v)


class EarlyStopping:
    # YOLOv5 simple early stopper
    def __init__(self, patience=30):
        self.best_fitness = 0.0  # i.e. mAP
        self.best_epoch = 0
        self.patience = patience or float('inf')  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch, fitness):
        if fitness >= self.best_fitness:  # >= 0 to allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        if stop:
            LOGGER.info(f'Stopping training early as no improvement observed in last {self.patience} epochs. '
                        f'Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n'
                        f'To update EarlyStopping(patience={self.patience}) pass a new patience value, '
                        f'i.e. `python train.py --patience 300` or use `--patience 0` to disable EarlyStopping.')
        return stop


# 常见的提高模型的鲁棒性的增强trock，被广泛的使用，全名：模型的指数加权平均方法
class ModelEMA:
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        """
        train.py
        model:
        decay:衰减函数参数，默认0.9999，考虑过去10000次的真实值
        :param model:
        :param decay:
        :param tau:
        :param updates:
        """
        # 创建ema模型
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        # 所有参数取消设置梯度
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            # msd:模型配置的字典，model state_dict,msd中的数据保持不变，用于训练
            msd = de_parallel(model).state_dict()  # model state_dict
            # 遍历模型配置字典
            for k, v in self.ema.state_dict().items():
                # 这里得到的v:预测值
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes，从model中复制相关属性值到self.ema中
        copy_attr(self.ema, model, include, exclude)
