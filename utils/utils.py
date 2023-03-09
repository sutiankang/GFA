import io
import time
import datetime
import torch
import numpy as np
import torch.distributed as dist
from collections import defaultdict, deque
from timm.utils import ModelEma
from datasets.dataset import UVOSDataset
from .distributed import is_dist_avail_and_initialized, is_main_process


def get_size(img_size):
    if isinstance(img_size, int):
        size = (img_size, img_size)
    elif isinstance(img_size, (list, tuple)) and len(img_size) == 1:
        size = (img_size[0], img_size[0])
    else:
        assert len(img_size) == 2, f"image size: {img_size} > 2 and is not a image"
        size = img_size
    return size


def create_dataset(args, dataset_names, is_train):
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    elif isinstance(dataset_names, list):
        dataset_names = dataset_names
    else:
        raise NotImplementedError
    dataset = UVOSDataset(args, is_train=is_train, datasets=dataset_names)

    return dataset


def create_model(args):

    if args.model == "segformer_b0_ade":
        from models.segformer.segformer import segformer_b0_ade
        if args.pretrained:
            pretrained = "your/pretrained/path"
            model = segformer_b0_ade(pretrained=pretrained, num_classes=1,
                                     uncertainty_probability=args.uncertainty_probability)
        else:
            model = segformer_b0_ade(num_classes=1, uncertainty_probability=args.uncertainty_probability)
            print("No use pretrained weights.")

    elif args.model == "segformer_b1_ade":
        from models.segformer.segformer import segformer_b1_ade
        if args.pretrained:
            pretrained = "your/pretrained/path"
            model = segformer_b1_ade(pretrained=pretrained, num_classes=1,
                                     uncertainty_probability=args.uncertainty_probability)
        else:
            model = segformer_b1_ade(num_classes=1, uncertainty_probability=args.uncertainty_probability)
            print("No use pretrained weights.")

    elif args.model == "segformer_b2_ade":
        from models.segformer.segformer import segformer_b2_ade
        if args.pretrained:
            pretrained = "your/pretrained/path"
            model = segformer_b2_ade(pretrained=pretrained, num_classes=1,
                                     uncertainty_probability=args.uncertainty_probability)
        else:
            model = segformer_b2_ade(num_classes=1, uncertainty_probability=args.uncertainty_probability)
            print("No use pretrained weights.")

    elif args.model == "segformer_b3_ade":
        from models.segformer.segformer import segformer_b3_ade
        if args.pretrained:
            pretrained = "your/pretrained/path"
            model = segformer_b3_ade(pretrained=pretrained, num_classes=1,
                                     uncertainty_probability=args.uncertainty_probability)
        else:
            model = segformer_b3_ade(num_classes=1, uncertainty_probability=args.uncertainty_probability)
            print("No use pretrained weights.")

    elif args.model == "segformer_b4_ade":
        from models.segformer.segformer import segformer_b4_ade
        if args.pretrained:
            pretrained = "your_pretrained/path"
            model = segformer_b4_ade(pretrained=pretrained, num_classes=1,
                                     uncertainty_probability=args.uncertainty_probability)
        else:
            model = segformer_b4_ade(num_classes=1, uncertainty_probability=args.uncertainty_probability)
            print("No use pretrained weights.")

    elif args.model == "segformer_b5_ade":
        from models.segformer.segformer import segformer_b5_ade
        if args.pretrained:
            pretrained = "your/pretrained/path"
            model = segformer_b5_ade(pretrained=pretrained, num_classes=1,
                                     uncertainty_probability=args.uncertainty_probability)
        else:
            model = segformer_b5_ade(num_classes=1, uncertainty_probability=args.uncertainty_probability)
            print("No use pretrained weights.")

    elif args.model == "segformer_b0_city":
        from models.segformer.segformer import segformer_b0_city
        if args.pretrained:
            pretrained = "your/pretrained/path"
            model = segformer_b0_city(pretrained=pretrained, num_classes=1,
                                      uncertainty_probability=args.uncertainty_probability)
        else:
            model = segformer_b0_city(num_classes=1, uncertainty_probability=args.uncertainty_probability)
            print("No use pretrained weights.")

    elif args.model == "segformer_b1_city":
        from models.segformer.segformer import segformer_b1_city
        if args.pretrained:
            pretrained = "your/pretrained/path"
            model = segformer_b1_city(pretrained=pretrained, num_classes=1,
                                      uncertainty_probability=args.uncertainty_probability)
        else:
            model = segformer_b1_city(num_classes=1, uncertainty_probability=args.uncertainty_probability)
            print("No use pretrained weights.")

    elif args.model == "segformer_b2_city":
        from models.segformer.segformer import segformer_b2_city
        if args.pretrained:
            pretrained = "your/pretrained/path"
            model = segformer_b2_city(pretrained=pretrained, num_classes=1,
                                      uncertainty_probability=args.uncertainty_probability)
        else:
            model = segformer_b2_city(num_classes=1, uncertainty_probability=args.uncertainty_probability)
            print("No use pretrained weights.")

    elif args.model == "segformer_b3_city":
        from models.segformer.segformer import segformer_b3_city
        if args.pretrained:
            pretrained = "your/pretrained/path"
            model = segformer_b3_city(pretrained=pretrained, num_classes=1,
                                      uncertainty_probability=args.uncertainty_probability)
        else:
            model = segformer_b3_city(num_classes=1, uncertainty_probability=args.uncertainty_probability)
            print("No use pretrained weights.")

    elif args.model == "segformer_b4_city":
        from models.segformer.segformer import segformer_b4_city
        if args.pretrained:
            pretrained = "your/pretrained/path"
            model = segformer_b4_city(pretrained=pretrained, num_classes=1,
                                      uncertainty_probability=args.uncertainty_probability)
        else:
            model = segformer_b4_city(num_classes=1, uncertainty_probability=args.uncertainty_probability)
            print("No use pretrained weights.")

    elif args.model == "segformer_b5_city":
        from models.segformer.segformer import segformer_b5_city
        if args.pretrained:
            pretrained = "your/pretrained/path"
            model = segformer_b5_city(pretrained=pretrained, num_classes=1,
                                      uncertainty_probability=args.uncertainty_probability)
        else:
            model = segformer_b5_city(num_classes=1, uncertainty_probability=args.uncertainty_probability)
            print("No use pretrained weights.")

    elif args.model == "segformer_b0":
        from models.segformer.segformer import segformer_b0
        if args.pretrained:
            pretrained = "your/pretrained/path"
            model = segformer_b0(pretrained=pretrained, num_classes=1,
                                 uncertainty_probability=args.uncertainty_probability)
        else:
            model = segformer_b0(num_classes=1, uncertainty_probability=args.uncertainty_probability)
            print("No use pretrained weights.")

    elif args.model == "segformer_b1":
        from models.segformer.segformer import segformer_b1
        if args.pretrained:
            pretrained = "your/pretrained/path"
            model = segformer_b1(pretrained=pretrained, num_classes=1,
                                 uncertainty_probability=args.uncertainty_probability)
        else:
            model = segformer_b1(num_classes=1, uncertainty_probability=args.uncertainty_probability)
            print("No use pretrained weights.")

    elif args.model == "segformer_b2":
        from models.segformer.segformer import segformer_b2
        if args.pretrained:
            pretrained = "your/pretrained/path"
            model = segformer_b2(pretrained=pretrained, num_classes=1,
                                 uncertainty_probability=args.uncertainty_probability)
        else:
            model = segformer_b2(num_classes=1, uncertainty_probability=args.uncertainty_probability)
            print("No use pretrained weights.")

    elif args.model == "segformer_b3":
        from models.segformer.segformer import segformer_b3
        if args.pretrained:
            pretrained = "your/pretrained/path"
            model = segformer_b3(pretrained=pretrained, num_classes=1,
                                 uncertainty_probability=args.uncertainty_probability)
        else:
            model = segformer_b3(num_classes=1, uncertainty_probability=args.uncertainty_probability)
            print("No use pretrained weights.")

    elif args.model == "segformer_b4":
        from models.segformer.segformer import segformer_b4
        if args.pretrained:
            pretrained = "your/pretrained/path"
            model = segformer_b4(pretrained=pretrained, num_classes=1,
                                 uncertainty_probability=args.uncertainty_probability)
        else:
            model = segformer_b4(num_classes=1, uncertainty_probability=args.uncertainty_probability)
            print("No use pretrained weights.")

    elif args.model == "segformer_b5":
        from models.segformer.segformer import segformer_b5
        if args.pretrained:
            pretrained = "your/pretrained/path"
            model = segformer_b5(pretrained=pretrained, num_classes=1,
                                 uncertainty_probability=args.uncertainty_probability)
        else:
            model = segformer_b5(num_classes=1, uncertainty_probability=args.uncertainty_probability)
            print("No use pretrained weights.")

    else:
        raise ValueError(f"Not support this model: {args.model}")

    return model


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def get_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                            norm_type)
    return total_norm


class Ensemble(object):
    def __init__(self,model, weights, map_location="cpu"):
        self.model = model
        self.weights = weights if isinstance(weights, list) else [weights]
        self.map_location = map_location

    def __call__(self):

        use_ema = False
        state_dicts = []
        final_state_dict = {}
        for weight in self.weights:
            checkpoint = torch.load(weight, map_location=self.map_location)
            if checkpoint.get("model_ema"):
                model_ema = ModelEma(
                    self.model,
                    decay=0.99996)
                use_ema = True
            state_dict = checkpoint["model_ema" if checkpoint.get("model_ema") else "model"]
            state_dicts.append(state_dict)
        for key in state_dicts[0].keys():
            temp_value = []
            for i in range(len(state_dicts)):
                temp_value.append(state_dicts[i][key].float())
            final_state_dict[key] = torch.stack(temp_value).mean(0)
            # final_state_dict[key] = torch.stack(temp_value).max(0)[0]
        if use_ema:
            final_state_dict = {"state_dict_ema": final_state_dict}
            _load_checkpoint_for_ema(model_ema, final_state_dict)
            self.model = model_ema.ema
        else:
            self.model.load_state_dict(final_state_dict, strict=True)
        return self.model


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, logger, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if (i + 1) % print_freq == 0 or i == len(iterable) - 1 and is_main_process():
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available() and is_main_process():
                    logger.info(log_msg.format(
                        i + 1, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    if is_main_process():
                        logger.info(log_msg.format(
                            i + 1, len(iterable), eta=eta_string,
                            meters=str(self),
                            time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if is_main_process():
            logger.info('{} Total time: {} ({:.4f} s / it)'.format(
                        header, total_time_str, total_time / len(iterable)))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.val

    @property
    def average(self):
        return np.round(self.avg, 5)