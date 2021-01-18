import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import os


class FeatureNet(nn.Module):
    def __init__(self, feat_size, use_feature=False):
        super(FeatureNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3, 2)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3, 2)
        self.batchnorm4 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*13*13, feat_size)  # 6*6 from image dimension
        self.fc2 = nn.Linear(feat_size, 14)
        self.use_feature = use_feature 
        
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        if self.use_feature:
            return x
        else:
            return self.fc2(x)

class FeatureNet2(nn.Module):
    def __init__(self, feat_size, use_feature=False):
        super(FeatureNet2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3, 2)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3, 2)
        self.fc1 = nn.Linear(64*13*13, 256)  # 6*6 from image dimension
        self.fc2 = nn.Linear(256, feat_size)
        self.fc3 = nn.Linear(feat_size, 14)
        self.use_feature = use_feature

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.use_feature:
            return x
        else:
            return self.fc3(x)

class FeatureNetC(nn.Module):
    def __init__(self, feat_size, train_mode=False):
        super(FeatureNetC, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3, 2)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3, 2)
        self.conv5 = nn.Conv2d(64, 128, 3, 2)
        self.fc1 = nn.Linear(128*6*6, 512)  # 6*6 from image dimension
        self.fc2 = nn.Linear(512, feat_size)
        self.fc3 = nn.Linear(feat_size, 14)
        self.train_mode = train_mode

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if self.train_mode:
            return x, self.fc3(x)
        else:
            return x
        
    
class PredictorBig(nn.Module):
    def __init__(self, feat_size):
        super(PredictorBig, self).__init__()
        self.feat_net1 = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2),
            nn.LeakyReLU(),
        )
        self.feat_net2 = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2),
            nn.LeakyReLU(),
        )
        self.fc1 = nn.Linear(128*13*13*2, 1024)  # 6*6 from image dimension
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, feat_size)

    def forward(self, x):
        # Input a pair of frames as a tensor (batch_size, 2, 1, 64, 64)
        x1 = self.feat_net1(x[:, 0])
        x2 = self.feat_net2(x[:, 1])
        x = torch.cat([x1, x2], dim=1)
        x = x.view(x.shape[0], -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x)
    
class PredictorMedium(nn.Module):
    def __init__(self, feat_size):
        super(PredictorMedium, self).__init__()
        self.feat_net = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3, 2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2),
            nn.LeakyReLU(),
        )
        self.fc1 = nn.Linear(128*13*13*2, 1024)  # 6*6 from image dimension
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, feat_size)

    def forward(self, x):
        # Input a pair of frames as a tensor (batch_size, 2, 1, 64, 64)
        x1 = self.feat_net(x[:, 0])
        x2 = self.feat_net(x[:, 1])
        x = torch.cat([x1, x2], dim=1)
        x = x.view(x.shape[0], -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x)
    
predictors = {'big': PredictorBig, 'medium': PredictorMedium}



class Sampler:
    def __init__(self, device, batch_size=100):
        self.batch_size = batch_size
        self.model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pretrained/smnist_gen.pth')
        self.n_past = 2

        # ---------------- load the models  ----------------
        tmp = torch.load(self.model_path)
        self.frame_predictor = tmp['frame_predictor'].to(device)
        self.posterior = tmp['posterior'].to(device)
        self.prior = tmp['prior'].to(device)
        self.frame_predictor.eval()
        self.prior.eval()
        self.posterior.eval()
        self.encoder = tmp['encoder'].to(device)
        self.decoder = tmp['decoder'].to(device)
        self.encoder.train()
        self.decoder.train()
        
        self.prior.batch_size = self.batch_size
        self.frame_predictor.batch_size = self.batch_size
        self.posterior.batch_size = self.batch_size
        
        # ---------------- set the options ----------------
        self.last_frame_skip = tmp['opt'].last_frame_skip
        
    # Input an array of size [self.n_past, batch_size, 1, 64, 64]
    # Roll out possible futures for n_future steps, returns an array of size [n_sample, self.n_past+n_future, batch_size, 1, 64, 64]
    def simulate(self, x, n_sample, n_future):
        all_gen = []
        for s in range(n_sample):
            self.frame_predictor.hidden = [[item.to(x.device) for item in items] for items in self.frame_predictor.init_hidden()]

            self.posterior.hidden = [[item.to(x.device) for item in items] for items in self.posterior.init_hidden()]
            self.prior.hidden = [[item.to(x.device) for item in items] for items in self.prior.init_hidden()]
            x_in = x[0]
            all_gen.append([])
            all_gen[s].append(x_in)
            for i in range(1, self.n_past+n_future):
                h = self.encoder(x_in)
                if self.last_frame_skip or i < self.n_past:	
                    h, skip = h
                else:
                    h, _ = h
                if i < self.n_past:
                    h_target = self.encoder(x[i])[0]
                    z_t, _, _ = self.posterior(h_target)
                    self.prior(h)
                    self.frame_predictor(torch.cat([h, z_t], 1))
                    x_in = x[i]
                    all_gen[s].append(x_in)
                else:
                    z_t, _, _ = self.prior(h)
                    h = self.frame_predictor(torch.cat([h, z_t], 1))
                    x_in = self.decoder([h, skip])
                    all_gen[s].append(x_in)

        # Concate everything into one big tensor
        all_gen = [torch.stack(item) for item in all_gen]
        all_gen = torch.stack(all_gen)
        return all_gen
