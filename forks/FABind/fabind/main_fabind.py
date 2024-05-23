import numpy as np
import os
from tqdm.auto import tqdm

import torch

from data import get_data
from torch_geometric.loader import DataLoader
from utils.metrics import *
from utils.utils import *
from datetime import datetime
from utils.logging_utils import Logger
import sys
import argparse
from torch.utils.data import RandomSampler
import random
from torch_scatter import scatter_mean
from utils.metrics_to_tsb import metrics_runtime_no_prefix
from torch.utils.tensorboard import SummaryWriter
# from torch.nn.utils import clip_grad_norm_
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import set_seed

def Seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser(description='FABind model training.')

parser.add_argument("-m", "--mode", type=int, default=0,
                    help="mode specify the model to use.")
parser.add_argument("-d", "--data", type=str, default="0",
                    help="data specify the data to use. \
                    0 for re-docking, 1 for self-docking.")
parser.add_argument('--seed', type=int, default=42,
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
parser.add_argument("--pocket-cls-loss-func", type=str, default='bce')

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
parser.add_argument('--norm-type', type=str, default="per_sample", choices=['per_sample', '4_sample', 'all_sample'])

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
parser.add_argument("--train-ligand-torsion-noise", action='store_true', default=False)
parser.add_argument("--train-pred-pocket-noise", type=float, default=0.0)
parser.add_argument('--esm2-concat-raw', action='store_true', default=False)
args = parser.parse_args()

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision=args.mixed_precision)
set_seed(args.seed)
# Seed_everything(seed=args.seed)
pre = f"{args.resultFolder}/{args.exp_name}"

if accelerator.is_main_process:
    os.system(f"mkdir -p {pre}/models")

    if not args.disable_tensorboard:
        tsb_runtime_dir = f"{pre}/tsb_runtime"
        os.system(f"mkdir -p {tsb_runtime_dir}")
        train_writer = SummaryWriter(log_dir=f'{tsb_runtime_dir}/train')
        valid_writer = SummaryWriter(log_dir=f'{tsb_runtime_dir}/valid')
        test_writer = SummaryWriter(log_dir=f'{tsb_runtime_dir}/test')
        test_writer_use_predicted_pocket = SummaryWriter(log_dir=f'{tsb_runtime_dir}/test_use_predicted_pocket')

accelerator.wait_for_everyone()

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
logger = Logger(accelerator=accelerator, log_path=f'{pre}/{timestamp}.log')

logger.log_message(f"{' '.join(sys.argv)}")

# torch.set_num_threads(1)
# # ----------without this, I could get 'RuntimeError: received 0 items of ancdata'-----------
torch.multiprocessing.set_sharing_strategy('file_system')

# train, valid, test: only native pocket. train_after_warm_up, all_pocket_test include all other pockets(protein center and P2rank result)
if args.redocking:
    args.compound_coords_init_mode = "redocking"
elif args.redocking_no_rotate:
    args.redocking = True
    args.compound_coords_init_mode = "redocking_no_rotate"

train, valid, test= get_data(args, logger, addNoise=args.addNoise, use_whole_protein=args.use_whole_protein, compound_coords_init_mode=args.compound_coords_init_mode, pre=args.data_path)
logger.log_message(f"data point train: {len(train)}, valid: {len(valid)}, test: {len(test)}")
num_workers = 10

if args.sample_n > 0:
    sampler = RandomSampler(train, replacement=True, num_samples=args.sample_n)
    train_loader = DataLoader(train, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], sampler=sampler, pin_memory=False, num_workers=num_workers)
    # sampler_update = RandomSampler(train_update, replacement=True, num_samples=args.sample_n)
    # train_update_loader = DataLoader(train_update, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], sampler=sampler_update, pin_memory=False, num_workers=num_workers)
else:
    train_loader = DataLoader(train, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], shuffle=True, pin_memory=False, num_workers=num_workers)
    # train_update_loader = DataLoader(train_update, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], shuffle=True, pin_memory=False, num_workers=num_workers)

# sampler2 = RandomSampler(train_after_warm_up, replacement=True, num_samples=args.sample_n)
# train_after_warm_up_loader = DataLoader(train_after_warm_up, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], sampler=sampler2, pin_memory=False, num_workers=num_workers)
# valid_batch_size = test_batch_size = 4
valid_loader = DataLoader(valid, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], shuffle=False, pin_memory=False, num_workers=num_workers)
test_loader = DataLoader(test, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], shuffle=False, pin_memory=False, num_workers=num_workers)
# valid_update_loader = DataLoader(valid_update, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], shuffle=False, pin_memory=False, num_workers=num_workers)
# test_update_loader = DataLoader(test_update, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], shuffle=False, pin_memory=False, num_workers=num_workers)
# not used
# all_pocket_test_loader = DataLoader(all_pocket_test, batch_size=2, follow_batch=['x', 'compound_pair'], shuffle=False, pin_memory=False, num_workers=4)

# import model is put here due to an error related to torch.utils.data.ConcatDataset after importing torchdrug.
from models.model import *
device = 'cuda'

model = get_model(args, logger, device)
if args.optim == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif args.optim == "adamw":
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

last_epoch = -1
steps_per_epoch = len(train_loader)
total_training_steps = args.total_epochs * len(train_loader)
scheduler_warm_up = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.5,
    end_factor=1,
    total_iters=args.warmup_epochs * len(train_loader),
    last_epoch=last_epoch,
)
if args.lr_scheduler == "constant":
    scheduler_post = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=(args.total_epochs - args.warmup_epochs)*len(train_loader), last_epoch=last_epoch)
elif args.lr_scheduler == "poly_decay":
    scheduler_post = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=(args.total_epochs - args.warmup_epochs)*len(train_loader), last_epoch=last_epoch)
elif args.lr_scheduler == "exp_decay":
    scheduler_post = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995, last_epoch=last_epoch)
elif args.lr_scheduler == "cosine_decay":
    scheduler_post = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.total_epochs - args.warmup_epochs)*len(train_loader), eta_min=1e-5, last_epoch=last_epoch)
elif args.lr_scheduler == "cosine_decay_restart":
    scheduler_post = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=0.0001, last_epoch=last_epoch)

scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[scheduler_warm_up, scheduler_post],
    milestones=[args.warmup_epochs*len(train_loader)],
)

(
    model,
    optimizer,
    scheduler,
    train_loader,
) = accelerator.prepare(
    model, optimizer, scheduler, train_loader,
)

output_last_epoch_dir = f"{pre}/models/epoch_last"
if os.path.exists(output_last_epoch_dir) and os.path.exists(os.path.join(output_last_epoch_dir, "pytorch_model.bin")):
    # ckpt = os.path.join(args.resultFolder, args.exp_name, 'models', "epoch_last.pt")
    # model_ckpt, opt_ckpt, model_args, last_epoch = torch.load(ckpt)
    # model.load_state_dict(model_ckpt, strict=True)
    # optimizer.load_state_dict(opt_ckpt)
    accelerator.load_state(output_last_epoch_dir)
    last_epoch = round(scheduler.state_dict()['last_epoch'] / steps_per_epoch) - 1
    logger.log_message(f'Load model from epoch: {last_epoch}')

# TODO Future debug when needed
# if args.restart:
#     model_ckpt, opt_ckpt, model_args, last_epoch = torch.load(args.restart)
#     model.load_state_dict(model_ckpt, strict=True)
#     optimizer.load_state_dict(opt_ckpt)
#     last_epoch = -1
# elif args.reload:
#     model_ckpt, opt_ckpt, model_args, last_epoch = torch.load(args.reload)
#     model.load_state_dict(model_ckpt, strict=True)
#     optimizer.load_state_dict(opt_ckpt)

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


# metrics_list = []
# valid_metrics_list = []
# test_metrics_list = []
# test_metrics_stage2_list = []

best_auroc = 0
best_f1_1 = 0
epoch_not_improving = 0

logger.log_message(f"Total epochs: {args.total_epochs}")
logger.log_message(f"Total training steps: {total_training_steps}")

for epoch in range(last_epoch+1, args.total_epochs):
    model.train()
    
    y_list = []
    y_pred_list = []
    com_coord_list = []
    com_coord_pred_list = []
    rmsd_list = []
    rmsd_2A_list = []
    rmsd_5A_list = []
    centroid_dis_list = []
    centroid_dis_2A_list = []
    centroid_dis_5A_list = []
    pocket_coord_list = []
    pocket_coord_pred_list = []
    # pocket_coord_pred_for_update_list = []
    pocket_cls_list = []
    pocket_cls_pred_list = []
    pocket_cls_pred_round_list = []
    protein_len_list = []
    # pdb_list = []
    count = 0
    skip_count = 0
    batch_loss = 0.0
    batch_by_pred_loss = 0.0
    batch_distill_loss = 0.0
    com_coord_batch_loss = 0.0
    pocket_cls_batch_loss = 0.0
    pocket_coord_batch_loss = 0.0
    keepNode_less_5_count = 0

    if args.disable_tqdm:
        data_iter = train_loader
    else:
        data_iter = tqdm(train_loader, mininterval=args.tqdm_interval, disable=not accelerator.is_main_process)
    for batch_id, data in enumerate(data_iter, start=1):
        optimizer.zero_grad()

        # Denote num_atom as N, num_amino_acid_of_pocket as M, num_amino_acid_of_protein as L
        # com_coord_pred: [B x N, 3]
        # y_pred, y_pred_by_coord: [B, N x M]
        # pocket_cls_pred, protein_out_mask_whole: [B, L]
        # p_coords_batched_whole: [B, L, 3]
        # pred_pocket_center: [B, 3]
        com_coord_pred, compound_batch, y_pred, y_pred_by_coord, pocket_cls_pred, pocket_cls, protein_out_mask_whole, p_coords_batched_whole, pred_pocket_center, dis_map, keepNode_less_5 = model(data, train=True)
        # y = data.y
        if y_pred.isnan().any() or com_coord_pred.isnan().any() or pocket_cls_pred.isnan().any() or pred_pocket_center.isnan().any() or y_pred_by_coord.isnan().any():
            print(f"nan occurs in epoch {epoch}")
            continue
        com_coord = data.coords
        pocket_cls_loss = args.pocket_cls_loss_weight * pocket_cls_criterion(pocket_cls_pred, pocket_cls.float()) * (protein_out_mask_whole.numel() / protein_out_mask_whole.sum())
        pocket_coord_loss = args.pocket_distance_loss_weight * pocket_coord_criterion(pred_pocket_center, data.coords_center)
        contact_loss = args.pair_distance_loss_weight * criterion(y_pred, dis_map) if len(dis_map) > 0 else torch.tensor([0])
        contact_by_pred_loss = args.pair_distance_loss_weight * criterion(y_pred_by_coord, dis_map) if len(dis_map) > 0 else torch.tensor([0])
        contact_distill_loss = args.pair_distance_distill_loss_weight * criterion(y_pred_by_coord, y_pred) if len(y_pred) > 0 else torch.tensor([0])

        com_coord_loss = args.coord_loss_weight * com_coord_criterion(com_coord_pred, com_coord) if len(com_coord) > 0 else torch.tensor([0])
        
        
        sd = ((com_coord_pred.detach() - com_coord) ** 2).sum(dim=-1)
        rmsd = scatter_mean(sd, index=compound_batch, dim=0).sqrt().detach()

        centroid_pred = scatter_mean(src=com_coord_pred, index=compound_batch, dim=0)
        centroid_true = scatter_mean(src=com_coord, index=compound_batch, dim=0)
        centroid_dis = (centroid_pred - centroid_true).norm(dim=-1)
        
        loss = com_coord_loss + \
            contact_loss + contact_by_pred_loss + contact_distill_loss + \
            pocket_cls_loss + \
            pocket_coord_loss
        
        accelerator.backward(loss)
        if args.clip_grad:
            # clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()

        batch_loss += len(y_pred)*contact_loss.item()
        batch_by_pred_loss += len(y_pred_by_coord)*contact_by_pred_loss.item()
        batch_distill_loss += len(y_pred_by_coord)*contact_distill_loss.item()
        com_coord_batch_loss += len(com_coord_pred)*com_coord_loss.item()

        pocket_cls_batch_loss += len(pocket_cls_pred)*pocket_cls_loss.item()
        pocket_coord_batch_loss += len(pred_pocket_center)*pocket_coord_loss.item()

        keepNode_less_5_count += keepNode_less_5

        y_list.append(dis_map.detach())
        y_pred_list.append(y_pred.detach())
        com_coord_list.append(com_coord)
        com_coord_pred_list.append(com_coord_pred.detach())
        rmsd_list.append(rmsd.detach())
        rmsd_2A_list.append((rmsd.detach() < 2).float())
        rmsd_5A_list.append((rmsd.detach() < 5).float())
        centroid_dis_list.append(centroid_dis.detach())
        centroid_dis_2A_list.append((centroid_dis.detach() < 2).float())
        centroid_dis_5A_list.append((centroid_dis.detach() < 5).float())

        
        batch_len = protein_out_mask_whole.sum(dim=1).detach()
        protein_len_list.append(batch_len)
        pocket_coord_pred_list.append(pred_pocket_center.detach())
        pocket_coord_list.append(data.coords_center)
        # use hard to calculate acc and skip samples
        for i, j in enumerate(batch_len):
            count += 1
            pocket_cls_list.append(pocket_cls.detach()[i][:j])
            pocket_cls_pred_list.append(pocket_cls_pred.detach()[i][:j].sigmoid())
            pocket_cls_pred_round_list.append(pocket_cls_pred.detach()[i][:j].sigmoid().round().int())
            pred_index_bool = (pocket_cls_pred.detach()[i][:j].sigmoid().round().int() == 1)
            if pred_index_bool.sum() == 0: # all the prediction is False, skip
                skip_count += 1

        if batch_id % args.log_interval == 0:
            stats_dict = {}
            stats_dict['step'] = batch_id
            stats_dict['lr'] = optimizer.param_groups[0]['lr']
            stats_dict['contact_loss'] = contact_loss.item()
            stats_dict['contact_by_pred_loss'] = contact_by_pred_loss.item()
            stats_dict['contact_distill_loss'] = contact_distill_loss.item()
            stats_dict['com_coord_loss'] = com_coord_loss.item()
            stats_dict['pocket_cls_loss'] = pocket_cls_loss.item()
            stats_dict['pocket_coord_loss'] = pocket_coord_loss.item()
            logger.log_stats(stats_dict, epoch, args, prefix='train')

    y = torch.cat(y_list)
    y_pred = torch.cat(y_pred_list)
    # y, y_pred = accelerator.gather((y, y_pred))

    com_coord = torch.cat(com_coord_list)
    com_coord_pred = torch.cat(com_coord_pred_list)
    # com_coord, com_coord_pred = accelerator.gather((com_coord, com_coord_pred))

    rmsd = torch.cat(rmsd_list)
    rmsd_2A = torch.cat(rmsd_2A_list)
    rmsd_5A = torch.cat(rmsd_5A_list)
    # rmsd, rmsd_2A, rmsd_5A = accelerator.gather((rmsd, rmsd_2A, rmsd_5A))
    rmsd_25 = torch.quantile(rmsd, 0.25)
    rmsd_50 = torch.quantile(rmsd, 0.50)
    rmsd_75 = torch.quantile(rmsd, 0.75)

    centroid_dis = torch.cat(centroid_dis_list)
    centroid_dis_2A = torch.cat(centroid_dis_2A_list)
    centroid_dis_5A = torch.cat(centroid_dis_5A_list)
    # centroid_dis, centroid_dis_2A, centroid_dis_5A = accelerator.gather((centroid_dis, centroid_dis_2A, centroid_dis_5A))
    centroid_dis_25 = torch.quantile(centroid_dis, 0.25)
    centroid_dis_50 = torch.quantile(centroid_dis, 0.50)
    centroid_dis_75 = torch.quantile(centroid_dis, 0.75)

    pocket_cls = torch.cat(pocket_cls_list)
    pocket_cls_pred = torch.cat(pocket_cls_pred_list)
    pocket_cls_pred_round = torch.cat(pocket_cls_pred_round_list)
    pocket_coord_pred = torch.cat(pocket_coord_pred_list)
    pocket_coord = torch.cat(pocket_coord_list)
    protein_len = torch.cat(protein_len_list)


    # pocket_cls, pocket_cls_pred, pocket_cls_pred_round, pocket_coord_pred, pocket_coord, protein_len = accelerator.gather(
    #     (pocket_cls, pocket_cls_pred, pocket_cls_pred_round, pocket_coord_pred, pocket_coord, protein_len)
    # )


    # count *= accelerator.num_processes
    # skip_count *= accelerator.num_processes
    # batch_loss *= accelerator.num_processes
    # batch_by_pred_loss *= accelerator.num_processes
    # batch_distill_loss *= accelerator.num_processes
    # com_coord_batch_loss *= accelerator.num_processes
    # pocket_cls_batch_loss *= accelerator.num_processes
    # pocket_coord_batch_loss *= accelerator.num_processes
    # keepNode_less_5_count *= accelerator.num_processes

    pocket_cls_accuracy = (pocket_cls_pred_round == pocket_cls).sum().item() / len(pocket_cls_pred_round)
    

    metrics = {"samples": count, "skip_samples": skip_count, "keepNode < 5": keepNode_less_5_count}
    metrics.update({"contact_loss":batch_loss/len(y_pred), "contact_by_pred_loss":batch_by_pred_loss/len(y_pred), "contact_distill_loss": batch_distill_loss/len(y_pred)})
    metrics.update({"com_coord_huber_loss": com_coord_batch_loss/len(com_coord_pred)})
    metrics.update({"rmsd": rmsd.mean().item(), "rmsd < 2A": rmsd_2A.mean().item(), "rmsd < 5A": rmsd_5A.mean().item()})
    metrics.update({"rmsd 25%": rmsd_25.item(), "rmsd 50%": rmsd_50.item(), "rmsd 75%": rmsd_75.item()})
    metrics.update({"centroid_dis": centroid_dis.mean().item(), "centroid_dis < 2A": centroid_dis_2A.mean().item(), "centroid_dis < 5A": centroid_dis_5A.mean().item()})
    metrics.update({"centroid_dis 25%": centroid_dis_25.item(), "centroid_dis 50%": centroid_dis_50.item(), "centroid_dis 75%": centroid_dis_75.item()})

    metrics.update({"pocket_cls_bce_loss": pocket_cls_batch_loss / len(pocket_coord_pred)})
    metrics.update({"pocket_coord_mse_loss": pocket_coord_batch_loss / len(pocket_coord_pred)})
    metrics.update({"pocket_cls_accuracy": pocket_cls_accuracy})
    metrics.update(pocket_metrics(pocket_coord_pred, pocket_coord))
    
    # logger.log_message(f"epoch {epoch:<4d}, train, " + print_metrics(metrics))
    logger.log_stats(metrics, epoch, args, prefix="Train")
    if accelerator.is_main_process and not args.disable_tensorboard:
        metrics_runtime_no_prefix(metrics, train_writer, epoch)
    
    accelerator.wait_for_everyone()

    # metrics_list.append(metrics)
    # release memory
    y, y_pred = None, None
    com_coord, com_coord_pred = None, None
    rmsd, rmsd_2A, rmsd_5A = None, None, None
    centroid_dis, centroid_dis_2A, centroid_dis_5A = None, None, None
    pocket_cls, pocket_cls_pred, pocket_cls_pred_round, pocket_coord_pred, pocket_coord, protein_len = None, None, None, None, None, None

    model.eval()
    # TODO check and think
    # use_y_mask = args.use_equivalent_native_y_mask or args.use_y_mask
    use_y_mask = False

    logger.log_message(f"Begin validation")
    if accelerator.is_main_process:
        if not args.disable_validate:
            metrics = evaluate_mean_pocket_cls_coord_multi_task(accelerator, args, valid_loader, model, com_coord_criterion, criterion, pocket_cls_criterion, pocket_coord_criterion, args.relative_k,
                                                                device, pred_dis=pred_dis, use_y_mask=use_y_mask, stage=1)

            # valid_metrics_list.append(metrics)
            # logger.log_message(f"epoch {epoch:<4d}, valid, " + print_metrics(metrics))
            logger.log_stats(metrics, epoch, args, prefix="Valid")

            metrics_runtime_no_prefix(metrics, valid_writer, epoch)
    
    logger.log_message(f"Begin test")
    if accelerator.is_main_process:
        metrics = evaluate_mean_pocket_cls_coord_multi_task(accelerator, args, test_loader, accelerator.unwrap_model(model), com_coord_criterion, criterion, pocket_cls_criterion, pocket_coord_criterion, args.relative_k,
                                                            accelerator.device, pred_dis=pred_dis, use_y_mask=use_y_mask, stage=1)
        # test_metrics_list.append(metrics)
        # logger.log_message(f"epoch {epoch:<4d}, test,  " + print_metrics(metrics))
        logger.log_stats(metrics, epoch, args, prefix="Test")

        if not args.disable_tensorboard:
            metrics_runtime_no_prefix(metrics, test_writer, epoch)

        metrics = evaluate_mean_pocket_cls_coord_multi_task(accelerator, args, test_loader, accelerator.unwrap_model(model), com_coord_criterion, criterion, pocket_cls_criterion, pocket_coord_criterion, args.relative_k,
                                                            accelerator.device, pred_dis=pred_dis, use_y_mask=use_y_mask, stage=2)
        # test_metrics_stage2_list.append(metrics)
        # logger.log_message(f"epoch {epoch:<4d}, testp,  " + print_metrics(metrics))
        logger.log_stats(metrics, epoch, args, prefix="Test_pp")

        if not args.disable_tensorboard:
            metrics_runtime_no_prefix(metrics, test_writer_use_predicted_pocket, epoch)

        # ckpt = (model.state_dict(), optimizer.state_dict(), args, epoch)
        # torch.save(ckpt, f"{pre}/models/epoch_{epoch}.pt")
        # torch.save(ckpt, f"{pre}/models/epoch_last.pt")
        output_dir = f"{pre}/models/epoch_{epoch}"
        accelerator.save_state(output_dir=output_dir)
        accelerator.save_state(output_dir=output_last_epoch_dir)
    
    accelerator.wait_for_everyone()

