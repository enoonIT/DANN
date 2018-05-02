import argparse

import numpy as np
import os

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

from dataset import data_loader
from models.model import entropy_loss, Combo, deco_types, classifier_list, deco_modes
from train.optim import optimizer_list, Optimizers, get_optimizer_and_scheduler
import itertools
import random

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
    parser.add_argument('--data_aug_mode', default="train", choices=["train", "simple", simple_tuned, "office"])
    parser.add_argument('--source', default=[data_loader.mnist], choices=data_loader.dataset_list, nargs='+')
    parser.add_argument('--target', default=data_loader.mnist_m, choices=data_loader.dataset_list)
    parser.add_argument('--n_classes', default=10, type=int)
    parser.add_argument('--target_limit', type=int, default=None, help="Number of max samples in target")
    parser.add_argument('--source_limit', type=int, default=None, help="Number of max samples in each source")
    # losses
    parser.add_argument('--DANN_weight', default=1.0, type=float)
    parser.add_argument('--entropy_loss_weight', default=0.0, type=float, help="Entropy loss on target, default is 0")
    # deco
    parser.add_argument('--use_deco', action="store_true", help="If true use deco architecture")
    parser.add_argument('--train_deco_weight', default=True, type=bool, help="Train the deco weight (True by default)")
    parser.add_argument('--train_image_weight', default=False, type=bool,
                        help="Train the image weight (False by default)")
    parser.add_argument('--deco_no_residual', action="store_true", help="If set, no residual will be applied to DECO")
    parser.add_argument('--deco_blocks', default=4, type=int)
    parser.add_argument('--deco_kernels', default=64, type=int)
    parser.add_argument('--deco_block_type', default='basic', choices=deco_types.keys(),
                        help="Which kind of deco block to use")
    parser.add_argument('--deco_output_channels', type=int, default=3, help="3 or 1")
    parser.add_argument('--deco_mode', default="shared", choices=deco_modes.keys())
    parser.add_argument('--deco_tanh', action="store_true", help="If set, tanh will be applied to DECO output")
    parser.add_argument('--deco_pretrain', default=0, type=int, help="Number of epoch to pretrain DECO (default is 0)")
    parser.add_argument('--deco_no_pool', action="store_true")
    parser.add_argument('--deco_deconv', action="store_true")
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
    name = "%s_lr:%g_BS:%d_eps:%d_IS:%d_DW:%g_DA%s" % (args.optimizer, args.lr, args.batch_size, args.epochs,
                                                       args.image_size, args.DANN_weight, args.data_aug_mode)
    if args.source_limit:
        name += "_sL%d" % args.source_limit
    if args.target_limit:
        name += "_tL%d" % args.target_limit
    if args.keep_pretrained_fixed:
        name += "_freezeNet"
    if args.entropy_loss_weight > 0.0:
        name += "_entropy:%g" % args.entropy_loss_weight
    if args.use_deco:
        name += "_deco%d_%d_%s_%dc" % (
            args.deco_blocks, args.deco_kernels, args.deco_block_type, args.deco_output_channels)
        if args.deco_pretrain > 0:
            name += "_pretrain%d" % args.deco_pretrain
        if args.deco_mode != "shared":
            name += "_" + args.deco_mode
        if args.deco_no_residual:
            name += "_no_res"
        if args.deco_tanh:
            name += "_tanh"
        elif args.train_deco_weight or args.train_image_weight:
            name += "_train%s%sWeight" % (
                "Deco" if args.train_deco_weight else "", "Image" if args.train_image_weight else "")
    else:
        name += "_vanilla"
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
    channels = x.shape[1]
    s = x.shape[2]
    y = x.swapaxes(1, 3).reshape(3, s * 3, s, channels).swapaxes(1, 2).reshape(s * 3, s * 3, channels).squeeze()[
        np.newaxis, ...]
    return y


def get_folder_name(source, target, generalization):
    if generalization:
        folder_prefix = "gen_"
    else:
        folder_prefix = "da_"
    return folder_prefix + '-'.join(source) + "_" + target


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def do_pretraining(num_epochs, dataloader_source, dataloader_target, model, logger):
    for name, deco in model.get_decos():
        print("Pretraining " + name)
        pretrain_deco(num_epochs, dataloader_source, dataloader_target, deco, logger, name)


def pretrain_deco(num_epochs, dataloader_source, dataloader_target, model, logger, mode):
    optimizer, scheduler = get_optimizer_and_scheduler(Optimizers.adam.value, model, num_epochs, 0.001, False)
    loss_f = nn.MSELoss().cuda()
    for epoch in range(num_epochs):
        model.train()
        if len(dataloader_source) > len(dataloader_target):
            source_loader = dataloader_source
            target_loader = itertools.cycle(dataloader_target)
        else:
            source_loader = itertools.cycle(dataloader_source)
            target_loader = dataloader_target

        for i, (source_batches, target_data) in enumerate(zip(source_loader, target_loader)):
            scheduler.step()
            optimizer.zero_grad()
            source_loss = 0.0
            for v, source_data in enumerate(source_batches):
                s_img, _ = source_data
                img_in = Variable(s_img).cuda()
                out = model(img_in)
                loss = loss_f(out, img_in)
                loss.backward()
                source_loss += loss.data.cpu().numpy()

            # pretrain target deco only if needed
            target_loss = 0.0
            target_image, _ = target_data
            img_in = Variable(target_image).cuda()
            out = model(img_in)
            loss = loss_f(out, img_in)
            loss.backward()
            target_loss = loss.data.cpu().numpy()
            optimizer.step()
            if i == 0:
                source_images = Variable(s_img[:9], volatile=True).cuda()
                target_images = Variable(target_image[:9], volatile=True).cuda()
                source_images = model(source_images)
                target_images = model(target_images)
                logger.image_summary("reconstruction/%s/source" % mode, to_grid(to_np(source_images)), epoch)
                logger.image_summary("reconstruction/%s/target" % mode, to_grid(to_np(target_images)), epoch)

        print("%d/%d - Reconstruction loss source: %g, target %g" % (epoch, num_epochs, source_loss, target_loss))
        logger.scalar_summary("reconstruction/%s/source" % mode, source_loss, epoch)
        logger.scalar_summary("reconstruction/%s/target" % mode, target_loss, epoch)


def softmax_list(source_target_similarity):
    if len(source_target_similarity) == 0:
        return None
    total_sum = sum(source_target_similarity)
    return [v / total_sum for v in source_target_similarity]


def train_epoch(epoch, dataloader_source, dataloader_target, optimizer, model, logger, n_epoch, cuda,
                dann_weight, entropy_weight, scheduler, generalize):
    model.train()
    len_dataloader = min(len(dataloader_source), len(dataloader_target))
    data_sources_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)

    batch_idx = 0
    domain_error = 0
    # TODO count epochs on source
    past_source_target_similarity = None
    weight_sources = True
    if generalize:
        weight_sources = False
    while batch_idx < len_dataloader:
        try:
            scheduler.step_iter()
        except AttributeError:
            pass
        absolute_iter_count = batch_idx + epoch * len_dataloader
        p = float(absolute_iter_count) / n_epoch / len_dataloader
        lambda_val = 2. / (1. + np.exp(-10 * p)) - 1
        if domain_error > 3.0:
            print("Shutting down DANN gradient to avoid collapse (iter %d)" % absolute_iter_count)
            lambda_val = 0.0
        data_sources_batch = data_sources_iter.next()
        # process source datasets (can be multiple)
        err_s_label = 0.0
        err_s_domain = 0.0
        num_source_domains = len(data_sources_batch)
        if generalize:
            target_domain_label = -1
        else:
            target_domain_label = num_source_domains
        model.set_deco_mode("source")
        source_domain_losses = []
        observed_domain_losses = []
        source_target_similarity = []
        for v, source_data in enumerate(data_sources_batch):
            s_img, s_label = source_data
            class_loss, domain_loss, observation_loss, target_similarity = compute_batch_loss(cuda, lambda_val, model,
                                                                                              s_img, s_label, v, target_domain_label)
            if weight_sources and past_source_target_similarity is not None:
                class_loss = class_loss * Variable(torch.from_numpy(len(data_sources_batch) * past_source_target_similarity[v]), requires_grad=True).cuda()
            loss = class_loss + dann_weight * domain_loss + observation_loss
            loss.backward()
            # used for logging only
            err_s_label += class_loss.data.cpu().numpy()
            source_domain_losses.append(domain_loss.data.cpu().numpy())
            observed_domain_losses.append(observation_loss.data.cpu().numpy())
            source_target_similarity.append(target_similarity)
            err_s_domain += domain_loss.data.cpu().numpy()
        past_source_target_similarity = softmax_list(source_target_similarity)
        err_s_label = err_s_label / num_source_domains
        err_s_domain = err_s_domain / num_source_domains

        entropy_target = 0
        err_t_domain = 0
        domain_error = err_s_domain
        if generalize is False:
            # training model using target data
            model.set_deco_mode("target")
            t_img, _ = data_target_iter.next()
            entropy_target, target_domain_loss, observation_loss, _ = compute_batch_loss(cuda, lambda_val, model, t_img, None, target_domain_label,
                                                                                         target_domain_label)
            loss = entropy_weight * entropy_target * lambda_val + dann_weight * target_domain_loss + observation_loss
            loss.backward()
            err_t_domain = target_domain_loss.data.cpu().numpy()
            observed_domain_losses.append(observation_loss.data.cpu().numpy())
            domain_error = (err_s_domain + err_t_domain) / 2

        # err = dann_weight * err_t_domain + dann_weight * err_s_domain + err_s_label + entropy_weight * entropy_target * lambda_val
        optimizer.step()
        optimizer.zero_grad()

        # logging stuff
        if (batch_idx % (len_dataloader / 2 + 1)) == 0:
            logger.scalar_summary("loss/source", err_s_label, absolute_iter_count)
            logger.scalar_summary("loss/domain", domain_error, absolute_iter_count)
            logger.scalar_summary("loss/observer_domain", sum(observed_domain_losses) / len(observed_domain_losses), absolute_iter_count)
            # for k, val in enumerate(source_domain_losses):
            #     logger.scalar_summary("loss/domain_s%d" % k, val, absolute_iter_count)
            for k, val in enumerate(past_source_target_similarity):
                logger.scalar_summary("similarity/prob/%d" % k, val, absolute_iter_count)
            logger.scalar_summary("loss/entropy_target", entropy_target, absolute_iter_count)
            print('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
                  % (epoch, batch_idx, len_dataloader, err_s_label, err_s_domain, err_t_domain))
        batch_idx += 1

    # at then end of one training epoch, save statistics and images
    if isinstance(model, Combo):
        # get target images
        target_images = random_items(iter(dataloader_target))[0][0]
        target_images = Variable(target_images[:9], volatile=True).cuda()
        model.set_deco_mode("target")
        target_images = model.deco(target_images)
        logger.image_summary("images/target", to_grid(to_np(target_images)), epoch)
        sources = random_items(iter(dataloader_source))[0]
        model.set_deco_mode("source")
        for n, (s_img, _) in enumerate(sources):
            source_images = Variable(s_img[:9], volatile=True).cuda()
            source_images = model.deco(source_images)
            logger.image_summary("images/source_%d" % n, to_grid(to_np(source_images)), epoch)
        for name, deco in model.get_decos():
            logger.scalar_summary("aux/%s/deco_to_image_ratio" % name, deco.ratio.data.cpu()[0], epoch)
            logger.scalar_summary("aux/%s/deco_weight" % name, deco.deco_weight.data.cpu()[0], epoch)
            logger.scalar_summary("aux/%s/image_weight" % name, deco.image_weight.data.cpu()[0], epoch)

    logger.scalar_summary("aux/p", p, epoch)
    logger.scalar_summary("aux/lambda", lambda_val, epoch)


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


def compute_batch_loss(cuda, lambda_val, model, img, label, _domain_label, target_label):
    eps = 1e-4
    domain_label = torch.ones(img.shape[0]).long() * _domain_label
    if cuda:
        img = img.cuda()
        if label is not None: label = label.cuda()
        domain_label = domain_label.cuda()
    class_output, domain_output, observer_output = model(input_data=Variable(img), lambda_val=lambda_val, domain=_domain_label)
    # compute losses
    if label is not None:
        class_loss = F.cross_entropy(class_output, Variable(label))
    else:
        class_loss = entropy_loss(class_output)

    domain_loss = F.cross_entropy(domain_output, Variable(domain_label))
    observer_loss = F.cross_entropy(observer_output, Variable(domain_label))
    if target_label > 0:
        if target_label < observer_output.shape[1]:
            target_similarity = (F.softmax(observer_output, 1)[:, target_label].mean()).data.cpu().numpy()
    else:  # generalization
        target_similarity = 0.0
    return class_loss, domain_loss, observer_loss, target_similarity
