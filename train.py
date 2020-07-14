import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import visdom
from torch.utils.data import DataLoader, TensorDataset

from neuralsea import NeuralSEA

###############################################################################
# Train Settings
parser = argparse.ArgumentParser(description='NeuralSEA training')
parser.add_argument('--batch_size',
                    type=int,
                    default=100,
                    help='training batch size (default: 100)')
parser.add_argument('--valid_batch_size',
                    type=int,
                    default=50,
                    help='validating batch size (default: 50)')
parser.add_argument('--epochs',
                    type=int,
                    default=100,
                    help='number of epochs to train for (default: 100)')
parser.add_argument('--lr',
                    type=float,
                    default=1e-3,
                    help='learnig rate (default: 1e-3)')
parser.add_argument('--weight_decay',
                    type=float,
                    default=1e-6,
                    help='weight decay rate for AdamW (default: 1e-6)')
parser.add_argument('--warm_start',
                    type=str,
                    default='',
                    help='path to warm start model')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--visdom', action='store_true', help='use visdom?')
parser.add_argument('--env',
                    type=str,
                    default='NeuralSEA',
                    help='visdom environmnet')
parser.add_argument('--X_train',
                    type=str,
                    default='./data/X_train.npy',
                    help='path to X train set (default: ./data/X_train.npy)')
parser.add_argument('--y_train',
                    type=str,
                    default='./data/y_train.npy',
                    help='path to y train set (default: ./data/y_train.npy)')
parser.add_argument('--X_valid',
                    type=str,
                    default='./data/X_valid.npy',
                    help='path to X valid set (default: ./data/X_valid.npy)')
parser.add_argument('--y_valid',
                    type=str,
                    default='./data/y_valid.npy',
                    help='path to y valid set (default: ./data/y_valid.npy)')
parser.add_argument(
    '--threads',
    type=int,
    default=4,
    help='number of threads for data loader to use (default: 4)')
parser.add_argument(
    '--pth_dir',
    type=str,
    default='checkpoints',
    help='where to save model checkpoints (default: checkpoints)')
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help='Seed for reproducibility (default: 42)')
###############################################################################


def seed(s):
    ''' Seed for reproducibility '''

    np.random.seed(s)
    torch.manual_seed(s)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_visdom(env):
    ''' Setup Visdom '''

    vis = visdom.Visdom(env=env)

    loss_win = vis.line(X=np.array([1]),
                        Y=np.array([0]),
                        opts={
                            'title': 'Loss - Epoch',
                            'showlegend': True,
                        },
                        name='train')
    vis.line(
        X=np.array([1]),
        Y=np.array([0]),
        win=loss_win,
        update='new',
        name='valid',
    )

    acc_win = vis.line(X=np.array([1]),
                       Y=np.array([0]),
                       opts={
                           'title': 'Accuracy - Epoch',
                           'showlegend': True,
                       },
                       name='valid')

    return vis, loss_win, acc_win


def setup_device(use_cuda):
    ''' Setup device '''

    if use_cuda:
        if not torch.cuda.is_available():
            raise Exception('No GPU found, please run without: --cuda')

        torch.cuda.empty_cache()
        device = torch.device('cuda')

    else:
        device = torch.device('cpu')

    print('Device:', device)
    print('=' * 15)

    return device


def get_dataset_loader(X_path, y_path, batch_size, threads):
    ''' Get set loader'''

    print('Load set')
    print('=' * 15)
    dataset = TensorDataset(torch.from_numpy(np.load(X_path)),
                            torch.from_numpy(np.load(y_path)))
    dataset_loader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                num_workers=threads,
                                shuffle=True)

    return dataset_loader


def build_net(warm_start, device):
    ''' Build the network '''

    print('Building the network')
    if warm_start != '':
        print('Warm start from network at:', warm_start)
        print('=' * 15)
        net = torch.load(warm_start, map_location=device).type(torch.float)
    else:
        net = NeuralSEA().to(device, dtype=torch.float)

    print('=' * 30)
    print(net)
    print('=' * 30)

    return net


def get_objective():
    ''' Get Objective '''

    objective = nn.BCEWithLogitsLoss()
    print('Objective:', objective)

    return objective


def get_optimizer(params, lr, weight_decay):
    ''' Get Optimizer '''

    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    print('Optimizer:', optimizer)

    return optimizer


def train(net, epoch, train_set_loader, device, objective, optimizer):
    ''' Train step '''

    net.train()
    epoch_loss = 0.0

    for i, data in enumerate(train_set_loader, 1):
        input, target = (data[0].to(device, dtype=torch.float),
                         data[1].to(device, dtype=torch.float))

        optimizer.zero_grad()

        output = net(input)
        loss = objective(output, target)
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

        if i % 10000 == 0:
            print(f'===> Epoch[{epoch}]({i}/{len(train_set_loader)}): \
                    Loss: {round(loss.item(), 7)}')

    train_loss = round(epoch_loss / len(train_set_loader), 7)
    print(f'=====> Epoch {epoch} Completed: \
            Avg. Loss: {train_loss}')

    return train_loss


def validate(net, valid_set_loader, device, objective):
    ''' Validate step '''

    net.eval()
    avg_loss = 0.0
    avg_acc = 0.0

    with torch.no_grad():
        for data in valid_set_loader:
            input, target = (data[0].to(device, dtype=torch.float),
                             data[1].to(device, dtype=torch.float))

            output = net(input)

            loss = objective(output, target)
            avg_loss += loss.item()

            pred = torch.sigmoid(output).round()
            avg_acc += (torch.sum(target.view(-1) == pred.view(-1)).float() /
                        target.numel()).item()

    valid_loss = round(avg_loss / len(valid_set_loader), 7)
    valid_acc = round(avg_acc / len(valid_set_loader), 7)
    print(f'=======> Avg. Valid Loss: {valid_loss}\
            Avg. Valid Acc: {valid_acc}')

    return valid_loss, valid_acc


def checkpoint(net, pth_dir, epoch, valid_acc):
    ''' Checkpoint step '''

    if not os.path.exists(pth_dir):
        os.mkdir(pth_dir)

    path = os.path.join(pth_dir,
                        f'neuralsea-epoch-{epoch}-acc-{valid_acc}.pth')
    torch.save(net, path)
    print(f'Checkpoint saved to {path}')


if __name__ == '__main__':
    args = parser.parse_args()
    print(args, end='\n\n')

    seed(args.seed)

    if args.visdom:
        vis, loss_win, acc_win = setup_visdom(args.env)

    device = setup_device(args.cuda)

    train_set_loader = get_dataset_loader(args.X_train, args.y_train,
                                          args.batch_size, args.threads)
    valid_set_loader = get_dataset_loader(args.X_valid, args.y_valid,
                                          args.valid_batch_size, args.threads)

    net = build_net(args.warm_start, device)
    objective = get_objective()
    optimizer = get_optimizer(net.parameters(), args.lr, args.weight_decay)

    # RUN
    print()
    print('Training...')
    for epoch in range(1, args.epochs + 1):
        train_loss = train(net, epoch, train_set_loader, device, objective,
                           optimizer)
        valid_loss, valid_acc = validate(net, valid_set_loader, device,
                                         objective)

        checkpoint(net, args.pth_dir, epoch, valid_acc)
        print()

        if args.visdom:
            vis.line(X=np.array([epoch]),
                     Y=np.array([train_loss]),
                     win=loss_win,
                     update='append',
                     name='train')

            vis.line(X=np.array([epoch]),
                     Y=np.array([valid_loss]),
                     win=loss_win,
                     update='append',
                     name='valid')

            vis.line(X=np.array([epoch]),
                     Y=np.array([valid_acc]),
                     win=acc_win,
                     update='append',
                     name='valid')
