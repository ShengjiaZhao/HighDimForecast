import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
from dataset import *
from model import *
import seaborn as sns

import gc
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--log_root', type=str, default='/data/hdim-forecast/log')

parser.add_argument('--feat_size', type=int, default=128)
parser.add_argument('--n_sample', type=int, default=256)
parser.add_argument('--n_past', type=int, default=2)
parser.add_argument('--n_future', type=int, default=10)

# Modeling parameters
parser.add_argument('--predictor_model', type=str, default='big')
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=8)

# Run related parameters
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--run_label', type=int, default=0)
args = parser.parse_args()
device = torch.device('cuda:%d' % args.gpu)
args.device = device

while True:
    args.log_dir = '/data/hdim-forecast/log3/pred/model=%s-seq=%d/%d-ns=%d-feat_size=%d-bs=%d-lr=%.5f-run=%d' % \
        (args.predictor_model, args.n_past, args.n_future, args.n_sample, args.feat_size, 
         args.batch_size, args.learning_rate, args.run_label)
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)
        break
    args.run_label += 1
print("Run number = %d" % args.run_label)
writer = SummaryWriter(args.log_dir)
log_writer = open(os.path.join(args.log_dir, 'results.txt'), 'w')

start_time = time.time()
global_iteration = 0
random.seed(args.run_label)  # Set a different random seed for different run labels
torch.manual_seed(args.run_label)
    
def log_scalar(name, value, epoch):
    writer.add_scalar(name, value, epoch)
    log_writer.write('%f ' % value)
    
def message(epoch):
    print("Finished epoch %d, time elapsed %.1f" % (epoch, time.time() - start_time))
    
    
feat_model = FeatureNetC(args.feat_size)
feat_model.load_state_dict(torch.load('pretrained/representation-c-%d.pt' % args.feat_size), strict=False)
feat_model = feat_model.to(device)
feat_model.eval()


multi_dataset = MovingMNISTMulti(train=True, n_past=args.n_past, n_future=args.n_future, 
                                 n_sample=args.n_sample, deterministic=False, last_only=True)
multi_loader = DataLoader(multi_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.batch_size)

predictor = predictors[args.predictor_model](args.feat_size).to(device)
exp_optim = optim.Adam(predictor.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(exp_optim, 20, 0.9)

# Learn the conditional expectation
for epoch in range(2000):
    
    multi_dataset.set_nsample(4)
    for idx, data in enumerate(multi_loader):
        exp_optim.zero_grad()

        bx, by, bl = data
        bx = bx.to(device)

        actual_feat = feat_model(bx[:, :, -1].view(-1, 1, 64, 64)).view(args.batch_size, 4, args.feat_size).detach()
        actual_exp = actual_feat.mean(dim=1)
        pred_exp = predictor(bx[:, 0, 0:2])
        
        loss_l2 = (actual_exp - pred_exp).pow(2).mean()
        loss_l2.backward()

        writer.add_scalar('loss_l2', loss_l2, global_iteration)
        exp_optim.step()
        global_iteration += 1
            
    errors = []
    baseline_error = []
    num_elem = 0
    multi_dataset.set_nsample(args.n_sample)
    plt.figure(figsize=(20, 20))
    palette = sns.color_palette('hls', 4)
    with torch.no_grad():
        for idx, data in enumerate(multi_loader):
            bx, by, bl = data
            bx = bx.to(device)
            
            actual_feat = feat_model(bx[:, :, -1].view(-1, 1, 64, 64)).view(args.batch_size, args.n_sample, args.feat_size)
            actual_exp = actual_feat.mean(dim=1)
            pred_exp = predictor(bx[:, 0, 0:2])
            errors.append(actual_exp - pred_exp)
            
            baseline_error.append(actual_feat[:, :args.n_sample//2, :].mean(dim=1) - actual_feat[:, args.n_sample//2:, :].mean(dim=1))
            num_elem += args.batch_size
            if num_elem > 1000:
                break
            if idx < 4:
                for i in range(36):
                    plt.subplot(6, 6, i+1)
                    plt.hist(actual_feat[0, :, i].cpu().numpy(), bins=20, color=palette[idx], alpha=0.5)
                    plt.axvline(pred_exp[0, i], color=palette[idx])
                    plt.axvline(actual_exp[0, i], color=palette[idx], linestyle=':')
            elif idx == 4:
                os.makedirs(os.path.join(args.log_dir, 'plot'), exist_ok=True)
                plt.savefig(os.path.join(args.log_dir, 'plot', 'hist-%d.png' % (epoch // 10)))
                plt.close()
            
        errors = torch.cat(errors)
        baseline_error = torch.cat(baseline_error)
    writer.add_scalar('loss_exp_l1', errors.abs().mean(), global_iteration)
    writer.add_scalar('loss_exp_l1_base', baseline_error.abs().mean(), global_iteration)
    scheduler.step()
    message(epoch)

    if (epoch+1) % 10 == 0:
        torch.save(predictor.state_dict(), 'pretrained/predictor2_%d-%d-%s.pt' % (args.feat_size, args.n_future, args.predictor_model))