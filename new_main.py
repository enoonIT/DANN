import random

import time

import pickle
import torch.backends.cudnn as cudnn
import torch.utils.data

from dataset import data_loader
from logger import Logger
from models.model import get_net
from test import test
from train.optim import get_optimizer_and_scheduler
from train.utils import get_name, get_folder_name, ensure_dir, get_args, simple_tuned, TrainingHelper

args = get_args()
print(args)
manual_seed = random.randint(1, 1000)
run_name = get_name(args, manual_seed)
print("Working on " + run_name)
log_folder = "logs/"
if args.tmp_log:
    log_folder = "/tmp/"
folder_name = get_folder_name(args.source, args.target, args.generalization)
logger = Logger("{}/{}/{}".format(log_folder, folder_name, run_name))

model_root = 'models'

cuda = True
cudnn.benchmark = True
lr = args.lr
batch_size = args.batch_size
image_size = args.image_size
test_batch_size = 1000
if image_size > 100:
    test_batch_size = 256
n_epoch = args.epochs

source_dataset_names = args.source
target_dataset_name = args.target

random.seed(manual_seed)
torch.manual_seed(manual_seed)

if args.data_path is not None:
    print("Loading from custom folder " + args.data_path)
    data_loader.dataset_folder = args.data_path
data_loader.load_from_folder = args.from_folder
args.domain_classes = 1 + len(args.source)
dataloader_source = data_loader.get_dataloader(args.source, batch_size, image_size, args.data_aug_mode,
                                               args.source_limit)
dataloader_target = data_loader.get_dataloader(args.target, batch_size, image_size, args.data_aug_mode,
                                               args.target_limit)
print("Len source %d, len target %d" % (len(dataloader_source), len(dataloader_target)))

# load model
my_net = get_net(args)

# setup optimizer
optimizer, scheduler = get_optimizer_and_scheduler(args.optimizer, my_net, args.epochs, args.lr,
                                                   args.keep_pretrained_fixed)

if cuda:
    my_net = my_net.cuda()

training_helper = TrainingHelper(my_net, args, logger, cuda)
if args.deco_pretrain_epochs > 0:
    training_helper.do_pretraining(dataloader_source, dataloader_target)
start = time.time()
# training
if args.data_aug_mode == simple_tuned:
    tune_stats = True
else:
    tune_stats = False
for epoch in range(n_epoch):
    scheduler.step()
    logger.scalar_summary("aux/lr", scheduler.get_lr()[0], epoch)
    training_helper.train_epoch(epoch, dataloader_source, dataloader_target, optimizer, scheduler)
    my_net.set_deco_mode("source")
    for d, source in enumerate(source_dataset_names):
        s_acc = test(source, epoch, my_net, image_size, d, test_batch_size, limit=args.source_limit,
                     tune_stats=tune_stats)
        if len(source_dataset_names) == 1:
            source_name = "acc/source"
        else:
            source_name = "acc/source_%s" % source
        logger.scalar_summary(source_name, s_acc, epoch)
    my_net.set_deco_mode("target")
    t_acc = test(target_dataset_name, epoch, my_net, image_size, len(args.source), test_batch_size,
                 limit=args.target_limit, tune_stats=tune_stats)
    logger.scalar_summary("acc/target", t_acc, epoch)

save_path = '{}/{}/{}_{}'.format(model_root, folder_name, run_name, epoch)
model_path = save_path + ".pth"
args_path = save_path + ".pkl"
print("Network saved to {}".format(model_path))
ensure_dir(model_path)
torch.save(my_net, model_path)
with open(args_path, "wb") as args_file:
    pickle.dump(args, args_file)
print('done, it took %g' % (time.time() - start))
