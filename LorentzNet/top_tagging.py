from top import dataset
from models import psi, LorentzNet
import torch
from torch import nn, optim
import argparse, json, time
import utils
import numpy as np
import torch.distributed as dist
import random
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

parser = argparse.ArgumentParser(description='Top tagging')
parser.add_argument('--exp_name', type=str, default='', metavar='N',
                    help='experiment_name')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help = 'test best model')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',   
                    help='input batch size for training')
parser.add_argument('--num_train', type=int, default=-1, metavar='N',
                    help='number of training samples')
parser.add_argument('--epochs', type=int, default=35, metavar='N',
                    help='number of training epochs')
parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                    help='number of warm-up epochs')
parser.add_argument('--c_weight', type=float, default=5e-3, metavar='N',
                    help='weight of x model')                 
parser.add_argument('--seed', type=int, default=99, metavar='N',
                    help='random seed')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--val_interval', type=int, default=1, metavar='N',
                    help='how many epochs to wait before validation')
parser.add_argument('--datadir', type=str, default='./data/top', metavar='N',
                    help='data dir')
parser.add_argument('--logdir', type=str, default='./logs/top', metavar='N',
                    help='folder to output logs')
parser.add_argument('--dropout', type=float, default=0.2, metavar='N',
                    help='dropout probability')
parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                    help='learning rate')
parser.add_argument('--n_hidden', type=int, default=72, metavar='N',
                    help='dim of latent space')
parser.add_argument('--n_layers', type=int, default=6, metavar='N',
                    help='number of LGEBs')
parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                    help='number of workers for the dataloader')
parser.add_argument('--weight_decay', type=float, default=1e-2, metavar='N',
                    help='weight decay')
parser.add_argument('--include_x', action='store_true',
                    help='whether to include raw positions as model input')
# augmentation related
parser.add_argument('--aug_group', type=str, default=None,
                    help='path to saved Lie algebra')
parser.add_argument('--aug_var', type=float, default=1.0,
                    help='extent of augmentation')
# EMLP related
parser.add_argument('--group', type=str, default=None,
                    help='path to saved Lie algebra')

parser.add_argument('--local_rank', type=int, default=0)
# invariant feat
parser.add_argument('--metric', type=str, default=None,
                    help='path to G-invariant metric')

def run(epoch, loader, partition):
    if partition == 'train':
        train_sampler.set_epoch(epoch)
        ddp_model.train()
    else:
        ddp_model.eval()

    res = {'time':0, 'correct':0, 'loss': 0, 'counter': 0, 'acc': 0,
           'loss_arr':[], 'correct_arr':[],'label':[],'score':[]}

    tik = time.time()
    loader_length = len(loader)

    for i, data in enumerate(loader):
        if partition == 'train':
            optimizer.zero_grad()

        batch_size, n_nodes, _ = data['Pmu'].size()
        atom_positions = data['Pmu'].reshape(batch_size * n_nodes, -1).to(device, dtype)
        atom_mask = data['atom_mask'].reshape(batch_size * n_nodes, -1).to(device)
        edge_mask = data['edge_mask'].reshape(batch_size * n_nodes * n_nodes, -1).to(device)
        nodes = data['nodes'].view(batch_size * n_nodes, -1).to(device,dtype)
        nodes = psi(nodes)
        edges = [a.to(device) for a in data['edges']]
        label = data['is_signal'].to(device, dtype).long()

        pred = ddp_model(scalars=nodes, x=atom_positions, edges=edges, node_mask=atom_mask,
                         edge_mask=edge_mask, n_nodes=n_nodes)
        
        predict = pred.max(1).indices
        correct = torch.sum(predict == label).item()
        loss = loss_fn(pred, label)
        
        if partition == 'train':
            loss.backward()
            optimizer.step()
        elif partition == 'test': 
            # save labels and probilities for ROC / AUC
            score = torch.nn.functional.softmax(pred, dim = -1)
            res['label'].append(label)
            res['score'].append(score)

        res['time'] = time.time() - tik
        res['correct'] += correct
        res['loss'] += loss.item() * batch_size
        res['counter'] += batch_size
        res['loss_arr'].append(loss.item())
        res['correct_arr'].append(correct)

        if i != 0 and i % args.log_interval == 0:
            running_loss = sum(res['loss_arr'][-args.log_interval:])/len(res['loss_arr'][-args.log_interval:])
            running_acc = sum(res['correct_arr'][-args.log_interval:])/(len(res['correct_arr'][-args.log_interval:])*batch_size)
            avg_time = res['time']/res['counter'] * batch_size
            tmp_counter = utils.sum_reduce(res['counter'], device = device)
            tmp_loss = utils.sum_reduce(res['loss'], device = device) / tmp_counter
            tmp_acc = utils.sum_reduce(res['correct'], device = device) / tmp_counter
            if (args.local_rank == 0):
                print(">> %s \t Epoch %d/%d \t Batch %d/%d \t Loss %.4f \t Running Acc %.3f \t Total Acc %.3f \t Avg Batch Time %.4f" %
                     (partition, epoch + 1, args.epochs, i, loader_length, running_loss, running_acc, tmp_acc, avg_time))

    torch.cuda.empty_cache()
    # ---------- reduce -----------
    if partition == 'test':
        res['label'] = torch.cat(res['label']).unsqueeze(-1)
        res['score'] = torch.cat(res['score'])
        res['score'] = torch.cat((res['label'],res['score']),dim=-1)
    res['counter'] = utils.sum_reduce(res['counter'], device = device).item()
    res['loss'] = utils.sum_reduce(res['loss'], device = device).item() / res['counter']
    res['acc'] = utils.sum_reduce(res['correct'], device = device).item() / res['counter']
    return res

def train(res):
    ### training and validation
    for epoch in range(0, args.epochs):
        train_res = run(epoch, dataloaders['train'], partition='train')
        print("Time: train: %.2f \t Train loss %.4f \t Train acc: %.4f" % (train_res['time'],train_res['loss'],train_res['acc']))
        if epoch % args.val_interval == 0:
            if (args.local_rank == 0):
                torch.save(ddp_model.state_dict(), f"{args.logdir}/{args.exp_name}/checkpoint-epoch-{epoch}.pt")
            dist.barrier() # wait master to save model
            with torch.no_grad():
                val_res = run(epoch, dataloaders['valid'], partition='valid')
            if (args.local_rank == 0): # only master process save
                res['lr'].append(optimizer.param_groups[0]['lr'])
                res['train_time'].append(train_res['time'])
                res['val_time'].append(val_res['time'])
                res['train_loss'].append(train_res['loss'])
                res['train_acc'].append(train_res['acc'])
                res['val_loss'].append(val_res['loss'])
                res['val_acc'].append(val_res['acc'])
                res['epochs'].append(epoch)

                ## save best model
                if val_res['acc'] > res['best_val']:
                    print("New best validation model, saving...")
                    torch.save(ddp_model.state_dict(), f"{args.logdir}/{args.exp_name}/best-val-model.pt")
                    res['best_val'] = val_res['acc']
                    res['best_epoch'] = epoch

                print("Epoch %d/%d finished." % (epoch, args.epochs))
                print("Train time: %.2f \t Val time %.2f" % (train_res['time'], val_res['time']))
                print("Train loss %.4f \t Train acc: %.4f" % (train_res['loss'], train_res['acc']))
                print("Val loss: %.4f \t Val acc: %.4f" % (val_res['loss'], val_res['acc']))
                print("Best val acc: %.4f at epoch %d." % (res['best_val'],  res['best_epoch']))

                json_object = json.dumps(res, indent=4)
                with open(f"{args.logdir}/{args.exp_name}/train-result.json", "w") as outfile:
                    outfile.write(json_object)

        ## adjust learning rate
        if (epoch < 31):
            lr_scheduler.step(metrics=val_res['acc'])
        else:
            for g in optimizer.param_groups:
                g['lr'] = g['lr']*0.5

        dist.barrier() # syncronize

def test(res):
    ### test on best model
    best_model = torch.load(f"{args.logdir}/{args.exp_name}/best-val-model.pt", map_location=device)
    ddp_model.load_state_dict(best_model)
    with torch.no_grad():
        test_res = run(0, dataloaders['test'], partition='test')

    pred = [torch.zeros_like(test_res['score']) for _ in range(dist.get_world_size())]
    dist.all_gather(pred, test_res['score'] )
    pred = torch.cat(pred).cpu()

    if (args.local_rank == 0):
        np.save(f"{args.logdir}/{args.exp_name}/score.npy",pred)
        fpr, tpr, thres, eB, eS  = utils.buildROC(pred[...,0], pred[...,2])
        auc = utils.roc_auc_score(pred[...,0], pred[...,2])

        metric = {'test_loss': test_res['loss'], 'test_acc': test_res['acc'],
                  'test_auc': auc, 'test_1/eB_0.3':1./eB[0],'test_1/eB_0.5':1./eB[1]}
        res.update(metric)
        print("Test: Loss %.4f \t Acc %.4f \t AUC: %.4f \t 1/eB 0.3: %.4f \t 1/eB 0.5: %.4f"
               % (test_res['loss'], test_res['acc'], auc, 1./eB[0], 1./eB[1]))
        json_object = json.dumps(res, indent=4)
        with open(f"{args.logdir}/{args.exp_name}/test-result.json", "w") as outfile:
            outfile.write(json_object)

if __name__ == "__main__":
    ### initialize args
    args = parser.parse_args()
    utils.args_init(args)

    ### set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ### load augmentation
    if args.aug_group is not None:
        L = torch.load(args.aug_group)
    else:
        L = None
    aug = L is not None

    ### initialize cuda
    dist.init_process_group(backend='nccl')
    device = torch.device("cuda:{}".format(args.local_rank))
    dtype = torch.float32
    ### load invariant metric tensor
    A = None if args.metric is None else torch.load(args.metric).to(device)
    ### load data
    train_sampler, dataloaders = dataset.retrieve_dataloaders(
                                    args.batch_size,
                                    args.num_workers,
                                    num_train=args.num_train,
                                    datadir=args.datadir,
                                    aug=aug,
                                    aug_group=L,
                                    aug_var=args.aug_var)
    ### create model
    L = torch.load(args.group).detach().cpu().numpy() if args.group is not None else None
    model = LorentzNet(n_scalar = 2, n_hidden = args.n_hidden, n_class = 2,
                       dropout = args.dropout, n_layers = args.n_layers,
                       c_weight = args.c_weight, A=A, include_x=args.include_x)
    ### parallelize model
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    ddp_model = DistributedDataParallel(model, device_ids=[args.local_rank])

    ### print model and data information
    if (args.local_rank == 0):
        pytorch_total_params = sum(p.numel() for p in ddp_model.parameters())
        # i = 0
        # for n, p in ddp_model.named_parameters():
        #     print(f'#{i} {n}: {p.numel()}')
        #     i += 1
        print("Network Size:", pytorch_total_params)
        for (split, dataloader) in dataloaders.items():
            print(f" {split} samples: {len(dataloader.dataset)}")

    ### optimizer
    optimizer = optim.AdamW(ddp_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ### lr scheduler
    base_scheduler = CosineAnnealingWarmRestarts(optimizer, 4, 2, verbose = False)
    lr_scheduler = utils.GradualWarmupScheduler(optimizer, multiplier=1,
                                                warmup_epoch=args.warmup_epochs,
                                                after_scheduler=base_scheduler) ## warmup

    ### loss function
    loss_fn = nn.CrossEntropyLoss()

    ### initialize logs
    res = {'epochs': [], 'lr' : [],
           'train_time': [], 'val_time': [],  'train_loss': [], 'val_loss': [],
           'train_acc': [], 'val_acc': [], 'best_val': 0, 'best_epoch': 0}

    if not args.test_mode:
        ### training and testing
        train(res)
        test(res)
    else:
        ### only test on best model
        test(res)