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

class FeatureNetFirstMoment(nn.Module):
    def __init__(self, feat_dim=0):
        super(FeatureNetFirstMoment, self).__init__()
    
    def forward(self, x):
        first_order = x 
        return first_order
    
class FeatureNetSecondMoment(nn.Module):
    def __init__(self, feat_dim=0):
        super(FeatureNetMoment, self).__init__()
    
    def forward(self, x):
        first_order = x 
        second_order = x.unsqueeze(-1).repeat(1, 1, x.shape[1]) * x.unsqueeze(-2).repeat(1, x.shape[1], 1)
        return torch.cat([first_order, second_order.view(-1, x.shape[1] * x.shape[1])], dim=1)
        
        
    
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
    


'''Defines the neural network, loss function and metrics'''

class PredictorRecurrent(nn.Module):
    def __init__(self, params):
        '''
        We define a recurrent network that predicts the future values of a time-dependent variable based on
        past inputs and covariates.
        '''
        super(PredictorRecurrent, self).__init__()
        self.params = params
        self.lstm = nn.LSTM(input_size=params.x_dim+params.y_dim,
                            hidden_size=params.lstm_hidden_dim,
                            num_layers=params.lstm_layers,
                            dropout=params.lstm_dropout,
                            bias=True,
                            batch_first=False)
        
        # initialize LSTM forget gate bias to be 1 as recommanded by http://proceedings.mlr.press/v37/jozefowicz15.pdf
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

        self.fc1 = nn.Linear(params.lstm_hidden_dim * params.lstm_layers, 1024)
        self.drop1 = nn.Dropout(p=params.lstm_dropout)
        self.fc2 = nn.Linear(1024, params.feat_size)
        

    def forward(self, x, y_prev):
        '''
        Predict mu and sigma of the distribution for z_t.
        Args:
            x: ([batch_size, n_past, x_dim]), the input feature
            y: ([batch_size, n_past, y_dim]), the previous step label
        Returns:
            pred 
            mu ([batch_size, y_dim]): estimated mean of z_t
            sigma ([batch_size, y_dim]): estimated standard deviation of z_t
            hidden ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM h from time step t
            cell ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM c from time step t
        '''
        hidden, cell = self.init_hidden(x.shape[0]), self.init_cell(x.shape[0])
        
        lstm_input = torch.cat([x, y_prev], dim=-1)
        # print(lstm_input.shape)
        output, (hidden, cell) = self.lstm(lstm_input.permute(1, 0, 2), (hidden, cell))
        
        # print(output.shape, hidden.shape, cell.shape)
            
        # use h from all three layers to calculate mu and sigma
        hidden_permute = hidden.permute(1, 0, 2).contiguous().view(hidden.shape[1], -1) 
        # print(hidden_permute.shape)
        fc = self.drop1(F.leaky_relu(self.fc1(hidden_permute)))
        return self.fc2(fc)

    def init_hidden(self, input_size):
        return torch.zeros(self.params.lstm_layers, input_size, self.params.lstm_hidden_dim, device=self.params.device)

    def init_cell(self, input_size):
        return torch.zeros(self.params.lstm_layers, input_size, self.params.lstm_hidden_dim, device=self.params.device)

    

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

    
predictors = {'big': PredictorBig, 'medium': PredictorMedium, 'lstm': PredictorRecurrent}

class LSTMSampler(nn.Module):
    def __init__(self, params):
        '''
        We define a recurrent network that predicts the future values of a time-dependent variable based on
        past inputs and covariates.
        '''
        super(LSTMSampler, self).__init__()
        self.params = params
        self.lstm = nn.LSTM(input_size=params.x_dim+params.y_dim,
                            hidden_size=params.lstm_hidden_dim,
                            num_layers=params.lstm_layers,
                            bias=True,
                            batch_first=False)
        
        # initialize LSTM forget gate bias to be 1 as recommanded by http://proceedings.mlr.press/v37/jozefowicz15.pdf
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

        self.relu = nn.ReLU()
        self.distribution_mu = nn.Linear(params.lstm_hidden_dim * params.lstm_layers, params.y_dim)
        self.distribution_presigma = nn.Linear(params.lstm_hidden_dim * params.lstm_layers, params.y_dim)
        self.distribution_sigma = nn.Softplus()
        

    def forward(self, x, y_prev, hidden, cell):
        '''
        Predict mu and sigma of the distribution for z_t.
        Args:
            x: ([1, batch_size, x_dim]), the input feature
            y: ([1, batch_size, y_dim]), the previous step label
            hidden ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM h from time step t-1
            cell ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM c from time step t-1
        Returns:
            mu ([batch_size, y_dim]): estimated mean of z_t
            sigma ([batch_size, y_dim]): estimated standard deviation of z_t
            hidden ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM h from time step t
            cell ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM c from time step t
        '''
        lstm_input = torch.cat((x, y_prev), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # use h from all three layers to calculate mu and sigma
        hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1) 
        pre_sigma = self.distribution_presigma(hidden_permute)
        mu = self.distribution_mu(hidden_permute)
        sigma = self.distribution_sigma(pre_sigma)  # softplus to make sure standard deviation is positive
        return mu, sigma, hidden, cell

    def init_hidden(self, input_size):
        return torch.zeros(self.params.lstm_layers, input_size, self.params.lstm_hidden_dim, device=self.params.device)

    def init_cell(self, input_size):
        return torch.zeros(self.params.lstm_layers, input_size, self.params.lstm_hidden_dim, device=self.params.device)

    def sample(self, x, y_prev, num_steps):
        '''
        Predict mu and sigma of the distribution for z_t.
        Args:
            x: ([seq_len+num_steps, batch_size, x_dim]), the input feature
            y: ([seq_len, batch_size, y_dim]), the previous step label
        Returns:
            samples ([num_steps, batch_size, y_dim]) the generated samples
        '''
        # print(x.shape, y_prev.shape)
        samples = torch.zeros(num_steps, x.shape[1], self.params.y_dim, device=self.params.device)
        
        hidden, cell = self.init_hidden(x.shape[1]), self.init_cell(x.shape[1])
        # print(hidden.shape)
        for t in range(self.params.n_past):
            mu, sigma, hidden, cell = self(x[t:t+1, :], y_prev[t:t+1, :], hidden, cell)
        # print(mu.shape)
        for j in range(num_steps):
            
            gaussian = torch.distributions.normal.Normal(mu, sigma)
            samples[j, :, :] = gaussian.sample()  # not scaled
            mu, sigma, hidden, celll = self(x[self.params.n_past+j].unsqueeze(0), 
                                 samples[j].unsqueeze(0), 
                                 hidden, cell)
                        
        return samples