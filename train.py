import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import visdom
from torch.utils.data import DataLoader, TensorDataset

from neuralsea import BasicSEBlock, NeuralSEA

###############################################################################
# Train Settings
parser = argparse.ArgumentParser(description='NeuralSEA training')
parser.add_argument('--num_blocks',
                    type=int,
                    default=1,
                    help='number of blocks (default: 1)')
parser.add_argument('--batch_size',
                    type=int,
                    default=128,
                    help='training batch size (default: 128)')
parser.add_argument('--valid_batch_size',
                    type=int,
                    default=32,
                    help='validating batch size (default: 32)')
parser.add_argument('--epochs',
                    type=int,
                    default=100,
                    help='number of epochs to train for (default: 100)')
parser.add_argument('--lr',
                    type=float,
                    default=1e-3,
                    help='learnig rate (default: 1e-3)')
parser.add_argument('--patience',
                    type=int,
                    default=5,
                    help='ReduceLROnPlateau patience (default: 5)')
parser.add_argument('--warm_start',
                    type=str,
                    default='',
                    help='path to warm start model')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--visdom', action='store_true', help='use visdom?')
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

args = parser.parse_args()

print(args, end='\n\n')
###############################################################################

# Seed for reproducibility
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# CUDA availability check
if args.cuda and not torch.cuda.is_available():
    raise Exception('No GPU found, please run without: --cuda')

# Setup Visdom if requested
if args.visdom:
    vis = visdom.Visdom(env='NeuralSEA')

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
                           'title': 'Valid Accuracy - Epoch',
                       })

# Setup device
device = torch.device('cuda' if args.cuda else 'cpu')
print('Device:', device)
print('=' * 15)

# Load Train/Valid set
# Train
print('Load Train set')
print('=' * 15)
train_set = TensorDataset(torch.from_numpy(np.load(args.X_train)),
                          torch.from_numpy(np.load(args.y_train)))
train_set_loader = DataLoader(dataset=train_set,
                              batch_size=args.batch_size,
                              num_workers=args.threads,
                              shuffle=True)

# Valid
print('Load Valid set')
print('=' * 15)
valid_set = TensorDataset(torch.from_numpy(np.load(args.X_valid)),
                          torch.from_numpy(np.load(args.y_valid)))
valid_set_loader = DataLoader(dataset=valid_set,
                              batch_size=args.valid_batch_size,
                              num_workers=args.threads,
                              shuffle=True)

# Build the Model
print('Building the model')
if args.warm_start != '':
    print('Warm start from model at:', args.warm_start)
    print('=' * 15)
    net = torch.load(args.warm_start).to(device)
else:
    net = NeuralSEA(BasicSEBlock, args.num_blocks).to(device)

print('=' * 30)
print(net)
print('=' * 30)

# Objective
objective = nn.BCEWithLogitsLoss()
print('Objective:', objective)
# Optimizer
optimizer = optim.Adam(net.parameters(), lr=args.lr)
print('Optimizer:', optimizer)
# Scheduler
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                    patience=args.patience,
                                                    verbose=True)


# Train step
def train(epoch):
    net.train()
    epoch_loss = 0.0

    for i, data in enumerate(train_set_loader, 1):
        input, target = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        output = net(input)
        loss = objective(output, target)
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

        if i % 10000 == 0:
            print(f'===> Epoch[{epoch}]({i}/{len(train_set_loader)}): \
                    Loss: {round(loss.item(), 4)}')

    train_loss = round(epoch_loss / len(train_set_loader), 4)
    print(f'=====> Epoch {epoch} Completed: \
            Avg. Loss: {train_loss}')

    return train_loss


# Val step
def validate():
    net.eval()
    avg_loss = 0.0
    avg_acc = 0.0

    with torch.no_grad():
        for data in valid_set_loader:
            input, target = data[0].to(device), data[1].to(device)

            output = net(input)

            loss = objective(output, target)
            avg_loss += loss.item()

            avg_acc += (torch.sum(
                torch.prod(target == torch.sigmoid(output).round(),
                           dim=-1)).float() / input.shape[0]).item()

    valid_loss = round(avg_loss / len(valid_set_loader), 4)
    valid_acc = round(avg_acc / len(valid_set_loader), 4)
    print(f'=======> Avg. Valid Loss: {valid_loss}\
            Avg. Valid Acc: {valid_acc}')

    return valid_loss, valid_acc


# Checkpoint step
def checkpoint(epoch, valid_loss):
    if not os.path.exists(args.pth_dir):
        os.mkdir(args.pth_dir)

    path = os.path.join(args.pth_dir,
                        f'neuralsea-epoch-{epoch}-loss-{valid_loss}.pth')
    torch.save(net, path)
    print(f'Checkpoint saved to {path}')


# RUN
print()
print('Training...')
for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    valid_loss, valid_acc = validate()
    lr_scheduler.step(valid_loss)

    checkpoint(epoch, valid_loss)
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
                 update='append')
