import numpy as np
import os

import torch

from data import get_data
from torch_geometric.loader import DataLoader
from utils.metrics import *
from utils.utils import *
from datetime import datetime
from utils.logging_utils import Logger
import sys
import argparse
import random
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import set_seed
import shlex

parser = argparse.ArgumentParser(description='FABind model testing.')

parser.add_argument("-m", "--mode", type=int, default=0,
                    help="mode specify the model to use.")
parser.add_argument("-d", "--data", type=str, default="0",
                    help="data specify the data to use. \
                    0 for re-docking, 1 for self-docking.")
parser.add_argument('--seed', type=int, default=600,
                    help="seed to use.")
parser.add_argument("--gs-tau", type=float, default=1,
                    help="Tau for the temperature-based softmax.")
parser.add_argument("--gs-hard", action='store_true', default=False,
                    help="Hard mode for gumbel softmax.")
parser.add_argument("--batch_size", type=int, default=8,
                    help="batch size.")

parser.add_argument("--restart", type=str, default=None,
                    help="continue the training from the model we saved from scratch.")
parser.add_argument("--reload", type=str, default=None,
                    help="continue the training from the model we saved.")
parser.add_argument("--addNoise", type=str, default=None,
                    help="shift the location of the pocket center in each training sample \
                    such that the protein pocket encloses a slightly different space.")

pair_interaction_mask = parser.add_mutually_exclusive_group()
# use_equivalent_native_y_mask is probably a better choice.
pair_interaction_mask.add_argument("--use_y_mask", action='store_true', default=False,
                    help="mask the pair interaction during pair interaction loss evaluation based on data.real_y_mask. \
                    real_y_mask=True if it's the native pocket that ligand binds to.")
pair_interaction_mask.add_argument("--use_equivalent_native_y_mask", action='store_true', default=False,
                    help="mask the pair interaction during pair interaction loss evaluation based on data.equivalent_native_y_mask. \
                    real_y_mask=True if most of the native interaction between ligand and protein happen inside this pocket.")

parser.add_argument("--use_affinity_mask", type=int, default=0,
                    help="mask affinity in loss evaluation based on data.real_affinity_mask")
parser.add_argument("--affinity_loss_mode", type=int, default=1,
                    help="define which affinity loss function to use.")

parser.add_argument("--pred_dis", type=int, default=1,
                    help="pred distance map or predict contact map.")
parser.add_argument("--posweight", type=int, default=8,
                    help="pos weight in pair contact loss, not useful if args.pred_dis=1")

parser.add_argument("--relative_k", type=float, default=0.01,
                    help="adjust the strength of the affinity loss head relative to the pair interaction loss.")
parser.add_argument("-r", "--relative_k_mode", type=int, default=0,
                    help="define how the relative_k changes over epochs")

parser.add_argument("--resultFolder", type=str, default="./result",
                    help="information you want to keep a record.")
parser.add_argument("--label", type=str, default="",
                    help="information you want to keep a record.")

parser.add_argument("--use-whole-protein", action='store_true', default=False,
                    help="currently not used.")

parser.add_argument("--data-path", type=str, default="/PDBbind_data/pdbbind2020",
                    help="Data path.")
                    
parser.add_argument("--exp-name", type=str, default="",
                    help="data path.")

parser.add_argument("--tqdm-interval", type=float, default=0.1,
                    help="tqdm bar update interval")

parser.add_argument("--lr", type=float, default=0.0001)

parser.add_argument("--pocket-coord-huber-delta", type=float, default=3.0)

parser.add_argument("--coord-loss-function", type=str, default='SmoothL1', choices=['MSE', 'SmoothL1'])

parser.add_argument("--coord-loss-weight", type=float, default=1.0)
parser.add_argument("--pair-distance-loss-weight", type=float, default=1.0)
parser.add_argument("--pair-distance-distill-loss-weight", type=float, default=1.0)
parser.add_argument("--pocket-cls-loss-weight", type=float, default=1.0)
parser.add_argument("--pocket-distance-loss-weight", type=float, default=0.05)
parser.add_argument("--pocket-cls-loss-func", type=str, default='bce', choices=['bce', 'dice'])

# parser.add_argument("--warm-mae-thr", type=float, default=5.0)

parser.add_argument("--use-compound-com-cls", action='store_true', default=False,
                    help="only use real pocket to run pocket classification task")

parser.add_argument("--compound-coords-init-mode", type=str, default="pocket_center_rdkit",
                    choices=['pocket_center_rdkit', 'pocket_center', 'compound_center', 'perturb_3A', 'perturb_4A', 'perturb_5A', 'random'])

parser.add_argument('--trig-layers', type=int, default=1)

parser.add_argument('--distmap-pred', type=str, default='mlp',
                    choices=['mlp', 'trig'])
parser.add_argument('--mean-layers', type=int, default=3)
parser.add_argument('--n-iter', type=int, default=5)
parser.add_argument('--inter-cutoff', type=float, default=10.0)
parser.add_argument('--intra-cutoff', type=float, default=8.0)
parser.add_argument('--refine', type=str, default='refine_coord',
                    choices=['stack', 'refine_coord'])

parser.add_argument('--coordinate-scale', type=float, default=5.0)
parser.add_argument('--geometry-reg-step-size', type=float, default=0.001)
parser.add_argument('--lr-scheduler', type=str, default="constant", choices=['constant', 'poly_decay', 'cosine_decay', 'cosine_decay_restart', 'exp_decay'])

parser.add_argument('--add-attn-pair-bias', action='store_true', default=False)
parser.add_argument('--explicit-pair-embed', action='store_true', default=False)
parser.add_argument('--opm', action='store_true', default=False)

parser.add_argument('--add-cross-attn-layer', action='store_true', default=False)
parser.add_argument('--rm-layernorm', action='store_true', default=False)
parser.add_argument('--keep-trig-attn', action='store_true', default=False)

parser.add_argument('--pocket-radius', type=float, default=20.0)

parser.add_argument('--rm-LAS-constrained-optim', action='store_true', default=False)
parser.add_argument('--rm-F-norm', action='store_true', default=False)
parser.add_argument('--norm-type', type=str, default="all_sample", choices=['per_sample', '4_sample', 'all_sample'])

# parser.add_argument("--only-predicted-pocket-mae-thr", type=float, default=3.0)
parser.add_argument('--noise-for-predicted-pocket', type=float, default=5.0)
parser.add_argument('--test-random-rotation', action='store_true', default=False)

parser.add_argument('--random-n-iter', action='store_true', default=False)
parser.add_argument('--clip-grad', action='store_true', default=False)

# one batch actually contains 20000 samples, not the size of training set
parser.add_argument("--sample-n", type=int, default=0, help="number of samples in one epoch.")

parser.add_argument('--fix-pocket', action='store_true', default=False)
parser.add_argument('--pocket-idx-no-noise', action='store_true', default=False)
parser.add_argument('--ablation-no-attention', action='store_true', default=False)
parser.add_argument('--ablation-no-attention-with-cross-attn', action='store_true', default=False)

parser.add_argument('--redocking', action='store_true', default=False)
parser.add_argument('--redocking-no-rotate', action='store_true', default=False)

parser.add_argument("--pocket-pred-layers", type=int, default=1, help="number of layers for pocket pred model.")
parser.add_argument('--pocket-pred-n-iter', type=int, default=1, help="number of iterations for pocket pred model.")

parser.add_argument('--use-esm2-feat', action='store_true', default=False)
parser.add_argument("--center-dist-threshold", type=float, default=8.0)

parser.add_argument("--mixed-precision", type=str, default='no', choices=['no', 'fp16'])
parser.add_argument('--disable-tqdm', action='store_true', default=False)
parser.add_argument('--log-interval', type=int, default=100)
parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'adamw'])
parser.add_argument("--warmup-epochs", type=int, default=15,
                    help="used in combination with relative_k_mode.")
parser.add_argument("--total-epochs", type=int, default=400,
                    help="option to switch training data after certain epochs.")
parser.add_argument('--disable-validate', action='store_true', default=False)
parser.add_argument('--disable-tensorboard', action='store_true', default=False)
parser.add_argument("--hidden-size", type=int, default=256)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--stage-prob", type=float, default=0.5)
parser.add_argument("--pocket-pred-hidden-size", type=int, default=128)

parser.add_argument("--local-eval", action='store_true', default=False)
parser.add_argument("--ckpt", type=str, default='../checkpoints/pytorch_model.bin')

parser.add_argument("--train-ligand-torsion-noise", action='store_true', default=False)
parser.add_argument("--train-pred-pocket-noise", type=float, default=0.0)
parser.add_argument("--esm2-concat-raw", action='store_true', default=False)
args_new = parser.parse_args()

command = "main_fabind.py -d 0 -m 5 --batch_size 3 --label baseline --addNoise 5 --tqdm-interval 60 --use-compound-com-cls --distmap-pred mlp --n-iter 8 --mean-layers 4 --refine refine_coord --coordinate-scale 5 --geometry-reg-step-size 0.001 --rm-layernorm --add-attn-pair-bias --explicit-pair-embed --add-cross-attn-layer --noise-for-predicted-pocket 0.0 --clip-grad --random-n-iter --pocket-idx-no-noise --seed 128 --use-esm2-feat --pocket-pred-layers 1 --pocket-pred-n-iter 1 --center-dist-threshold 4 --pocket-cls-loss-func bce --mixed-precision no --disable-tqdm --disable-validate --log-interval 50 --optim adamw --norm-type per_sample --weight-decay 0.01 --hidden-size 512 --pocket-pred-hidden-size 128 --stage-prob 0.25"
command = shlex.split(command)

args = parser.parse_args(command[1:])
args.local_eval = args_new.local_eval
args.ckpt = args_new.ckpt
args.data_path = args_new.data_path
args.resultFolder = args_new.resultFolder
args.seed = args_new.seed
args.exp_name = args_new.exp_name
args.batch_size = args_new.batch_size
args.tqdm_interval = 0.1
args.disable_tqdm = False

set_seed(args.seed)

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision=args.mixed_precision)

pre = f"{args.resultFolder}/{args.exp_name}"

accelerator.wait_for_everyone()

os.makedirs(pre, exist_ok=True)
logger = Logger(accelerator=accelerator, log_path=f'{pre}/test.log')

logger.log_message(f"{' '.join(sys.argv)}")

# torch.set_num_threads(1)

torch.multiprocessing.set_sharing_strategy('file_system')

if args.redocking:
    args.compound_coords_init_mode = "redocking"
elif args.redocking_no_rotate:
    args.redocking = True
    args.compound_coords_init_mode = "redocking_no_rotate"

train, valid, test= get_data(args, logger, addNoise=args.addNoise, use_whole_protein=args.use_whole_protein, compound_coords_init_mode=args.compound_coords_init_mode, pre=args.data_path)
logger.log_message(f"data point train: {len(train)}, valid: {len(valid)}, test: {len(test)}")
num_workers = 10

test_loader = DataLoader(test, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], shuffle=False, pin_memory=False, num_workers=num_workers)
test_unseen_pdb_list = [line.strip() for line in open('split_pdb_id/unseen_test_index')]

test_unseen_index = test.data.query("(group =='test') and (pdb in @test_unseen_pdb_list)").index.values
# double check
test_unseen_index_for_select = np.array([np.where(test._indices == i) for i in test_unseen_index]).reshape(-1)
test_unseen = test.index_select(test_unseen_index_for_select)
test_unseen_loader = DataLoader(test_unseen, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], shuffle=False, pin_memory=False, num_workers=10)

from models.model import *
device = 'cuda'

model = get_model(args, logger, device)

model = accelerator.prepare(model)

model.load_state_dict(torch.load(args.ckpt))

if args.pred_dis:
    criterion = nn.MSELoss()
    pred_dis = True
else:
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.posweight))

if args.coord_loss_function == 'MSE':
    com_coord_criterion = nn.MSELoss()
elif args.coord_loss_function == 'SmoothL1':
    com_coord_criterion = nn.SmoothL1Loss()

if args.pocket_cls_loss_func == 'bce':
    pocket_cls_criterion = nn.BCEWithLogitsLoss(reduction='mean')

pocket_coord_criterion = nn.HuberLoss(delta=args.pocket_coord_huber_delta)

model.eval()

logger.log_message(f"Begin test")
if accelerator.is_main_process:
    metrics = evaluate_mean_pocket_cls_coord_multi_task(accelerator, args, test_loader, accelerator.unwrap_model(model), com_coord_criterion, criterion, pocket_cls_criterion, pocket_coord_criterion, args.relative_k,
                                                        accelerator.device, pred_dis=pred_dis, use_y_mask=False, stage=2)
    logger.log_stats(metrics, 0, args, prefix="Test_all")

    metrics = evaluate_mean_pocket_cls_coord_multi_task(accelerator, args, test_unseen_loader, accelerator.unwrap_model(model), com_coord_criterion, criterion, pocket_cls_criterion, pocket_coord_criterion, args.relative_k,
                                                        accelerator.device, pred_dis=pred_dis, use_y_mask=False, stage=2)
    logger.log_stats(metrics, 0, args, prefix="Test_unseen")
accelerator.wait_for_everyone()
