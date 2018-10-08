import argparse
import os
import random

import numpy as np
import torch.nn.functional as F
from torchvision.utils import make_grid

from dataset import data_loader
from models.model import entropy_loss, classifier_list
from train.optim import optimizer_list, Optimizers

simple_tuned = "simple-tuned"


def get_args():
    parser = argparse.ArgumentParser()
    # optimizer
    parser.add_argument('--optimizer', choices=optimizer_list, default=Optimizers.adam.value)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--keep_pretrained_fixed', action="store_true")
    # data
    parser.add_argument('--image_size', type=int, default=28)
    parser.add_argument('--data_aug_mode', default="train", choices=["train", "simple", simple_tuned, "office", "office-caffe"])
    parser.add_argument('--source', default=[data_loader.mnist], choices=data_loader.dataset_list, nargs='+')
    parser.add_argument('--target', default=data_loader.mnist_m, choices=data_loader.dataset_list)
    parser.add_argument('--n_classes', default=10, type=int)
    parser.add_argument('--target_limit', type=int, default=None, help="Number of max samples in target")
    parser.add_argument('--source_limit', type=int, default=None, help="Number of max samples in each source")
    # losses
    parser.add_argument('--entropy_loss_weight', default=0.0, type=float, help="Entropy loss on target, default is 0")
    # misc
    parser.add_argument('--suffix', help="Will be added to end of name", default="")
    parser.add_argument('--classifier', default=None, choices=classifier_list.keys())
    parser.add_argument('--tmp_log', action="store_true", help="If set, logger will save to /tmp instead")
    parser.add_argument('--generalization', action="store_true",
                        help="If set, the target will not be used during training")
    args = parser.parse_args()
    args.source = sorted(args.source)
    return args


def get_name(args, seed):
    name = "%s_lr:%g_BS:%d_eps:%d_IS:%d_DA%s" % (args.optimizer, args.lr, args.batch_size, args.epochs,
                                                       args.image_size, args.data_aug_mode)
    if args.source_limit:
        name += "_sL%d" % args.source_limit
    if args.target_limit:
        name += "_tL%d" % args.target_limit
    if args.keep_pretrained_fixed:
        name += "_freezeNet"
    if args.entropy_loss_weight > 0.0:
        name += "_entropy:%g" % args.entropy_loss_weight
    if args.classifier:
        name += "_" + args.classifier
    if args.generalization:
        name += "_generalization"
    if args.suffix:
        name += "_" + args.suffix
    return name + "_%d" % (seed)


def to_np(x):
    return x.data.cpu().numpy()


def to_grid(x):
    y = make_grid(x, nrow=3, padding=1, normalize=False, range=None, scale_each=False, pad_value=0)
    tmp = y.cpu().numpy().swapaxes(0, 1).swapaxes(1, 2)
    return tmp[np.newaxis, ...]


def get_folder_name(source, target):
    return '-'.join(source) + "_" + target


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def softmax_list(source_target_similarity):
    total_sum = sum(source_target_similarity)
    if total_sum == 0:
        total_sum = 1
    return [v / total_sum for v in source_target_similarity]


def train_epoch(epoch, dataloader_source, dataloader_target, optimizer, model, logger, n_epoch, cuda,
                entropy_weight, scheduler, generalize):
    model.train()
    len_dataloader = min(len(dataloader_source), len(dataloader_target))
    data_sources_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)

    batch_idx = 0

    while batch_idx < len_dataloader:
        try:
            scheduler.step_iter()
        except AttributeError:
            pass
        absolute_iter_count = batch_idx + epoch * len_dataloader

        data_sources_batch = data_sources_iter.next()
        # process source datasets (can be multiple)
        err_s_label = 0.0
        num_source_domains = len(data_sources_batch)

        for v, source_data in enumerate(data_sources_batch):
            s_img, s_label = source_data
            class_loss = compute_batch_loss(cuda, model, s_img, s_label, v)
            class_loss.backward()
            # used for logging only
            err_s_label += class_loss.data.cpu().numpy()

        err_s_label = err_s_label / num_source_domains

        entropy_target = 0
        if generalize is False:
            # training model using target data
            t_img, _ = data_target_iter.next()
            entropy_target = compute_batch_loss(cuda, model, t_img, None, num_source_domains)
            loss = entropy_weight * entropy_target
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # logging stuff
        if (batch_idx % (len_dataloader / 2 + 1)) == 0:
            logger.scalar_summary("loss/source", err_s_label, absolute_iter_count)
            logger.scalar_summary("loss/entropy_target", entropy_target, absolute_iter_count)
            print('epoch: %d, [iter: %d / all %d], err_s_label: %f, entropy_target: %f' % (epoch, batch_idx, len_dataloader, err_s_label, entropy_target))
        batch_idx += 1


# from https://code.activestate.com/recipes/426332-picking-random-items-from-an-iterator/
def random_items(iterable, k=1):
    result = [None] * k
    for i, item in enumerate(iterable):
        if i < k:
            result[i] = item
        else:
            j = int(random.random() * (i + 1))
            if j < k:
                result[j] = item
    random.shuffle(result)
    return result


def compute_batch_loss(cuda, model, img, label, domain_label):
    if cuda:
        img = img.cuda()
        if label is not None: label = label.cuda()
    class_output = model(input_data=img, domain=domain_label)
    # compute losses
    if label is not None:
        class_loss = F.cross_entropy(class_output, label)
    else:
        class_loss = entropy_loss(class_output)

    return class_loss
