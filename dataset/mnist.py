import socket
import numpy as np
from torchvision import datasets, transforms
import torch 
import os

max_s = 8

class MovingMNIST(object):
    
    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def __init__(self, train, n_past=2, n_future=10, num_digits=2, image_size=64, deterministic=False, last_only=True):
        self.n_past = n_past
        self.n_future = n_future
        self.num_digits = num_digits  
        self.image_size = image_size 
        self.step_length = 0.1
        self.digit_size = 32
        self.deterministic = deterministic
        self.last_only = last_only
        self.seed_is_set = False # multi threaded loading
        self.channels = 1 

        self.data = datasets.MNIST(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), 'MNIST'),
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.Scale(self.digit_size),
                 transforms.ToTensor()]))

        self.N = len(self.data) 

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
          
    def __len__(self):
        return self.N

    def __getitem__(self, index):
        if self.last_only:
            seq_len = self.n_past + 1
        else:
            seq_len = self.n_past + self.n_future
        self.set_seed(index)
        image_size = self.image_size
        digit_size = self.digit_size
        x = np.zeros((seq_len, self.channels, image_size, image_size), dtype=np.float32)
        y = np.zeros(self.num_digits, dtype=np.int32)
        loc = np.zeros((seq_len, self.num_digits, 2), dtype=np.float32)
        
        for n in range(self.num_digits):
            idx = np.random.randint(self.N)
            digit, label = self.data[idx]
            
            y[n] = label
            sx = np.random.randint(image_size-digit_size)
            sy = np.random.randint(image_size-digit_size)
            dx = np.random.randint(-max_s+1, max_s)
            dy = np.random.randint(-max_s+1, max_s)
            t_ = 0
            for t in range(seq_len):
                dx += np.random.randint(-1, 2)
                dy += np.random.randint(-1, 2)
                if sy < 0:
                    sy = 0 
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(1, max_s)
                        dx = np.random.randint(-max_s+1, max_s)
                elif sy >= image_size-32:
                    sy = image_size-32-1
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(-max_s+1, 0)
                        dx = np.random.randint(-max_s+1, max_s)
                    
                if sx < 0:
                    sx = 0 
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(1, max_s)
                        dy = np.random.randint(-max_s+1, max_s)
                elif sx >= image_size-32:
                    sx = image_size-32-1
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(-max_s+1, 0)
                        dy = np.random.randint(-max_s+1, max_s)
                
                if t < self.n_past or (not self.last_only) or (t == seq_len-1):
                    x[t_, :, sy:sy+32, sx:sx+32] += digit.numpy()
                    loc[t_, n, 0] = sx+16
                    loc[t_, n, 1] = sy+16
                    t_ += 1
                sy += dy
                sx += dx

        x[x>1] = 1.
        return x, y, (loc - 16) / 48.

    



class MovingMNISTMulti(object):
    
    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def __init__(self, train, n_past=2, n_future=10, n_sample=500, num_digits=2, image_size=64, deterministic=True, last_only=False):
        self.n_past = n_past
        self.n_future = n_future
        self.n_sample = n_sample
        
        self.num_digits = num_digits  
        self.image_size = image_size 
        self.step_length = 0.1
        self.digit_size = 32
        self.deterministic = deterministic
        self.seed_is_set = False # multi threaded loading
        self.channels = 1 
        self.last_only = last_only
        
        self.data = datasets.MNIST(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), 'MNIST'),
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.Scale(self.digit_size),
                 transforms.ToTensor()]))

        self.N = len(self.data) 

    def set_nsample(self, n_sample):
        self.n_sample = n_sample
        
    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
          
    def __len__(self):
        return self.N

    def __getitem__(self, index):
        if self.last_only:
            seq_len = self.n_past + 1
        else:
            seq_len = self.n_past + self.n_future
            
        self.set_seed(index)
        image_size = self.image_size
        digit_size = self.digit_size
        x = torch.zeros(self.n_sample, seq_len, self.channels, image_size, image_size, dtype=torch.float32)
        y = torch.zeros(self.num_digits, dtype=torch.int32)
        loc = torch.zeros(self.n_sample, seq_len, self.num_digits, 2, dtype=torch.float32)
        
        for n in range(self.num_digits):
            idx = np.random.randint(self.N)
            digit, label = self.data[idx]
            
            y[n] = label
            sx_init = np.random.randint(image_size-digit_size)
            sy_init = np.random.randint(image_size-digit_size)
            dx_init = np.random.randint(-max_s+1, max_s)
            dy_init = np.random.randint(-max_s+1, max_s)
            for rep in range(self.n_sample):
                sx, sy = sx_init, sy_init
                dx, dy = dx_init, dy_init
                t_ = 0
                for t in range(self.n_past+self.n_future):
                    if t < self.n_past: # Set a seed that doesn't depend on rep to ensure the past samples are identical
                        self.set_seed(index * 100000 + t)
                    else:
                        self.set_seed(index * 100000 + t * 1000 + rep)
                    dx += np.random.randint(-1, 2)
                    dy += np.random.randint(-1, 2)
                    if sy < 0:
                        sy = 0 
                        if self.deterministic:
                            dy = -dy
                        else:
                            dy = np.random.randint(1, max_s)
                            dx = np.random.randint(-max_s+1, max_s)
                    elif sy >= image_size-32:
                        sy = image_size-32-1
                        if self.deterministic:
                            dy = -dy
                        else:
                            dy = np.random.randint(-max_s+1, 0)
                            dx = np.random.randint(-max_s+1, max_s)

                    if sx < 0:
                        sx = 0 
                        if self.deterministic:
                            dx = -dx
                        else:
                            dx = np.random.randint(1, max_s)
                            dy = np.random.randint(-max_s+1, max_s)
                    elif sx >= image_size-32:
                        sx = image_size-32-1
                        if self.deterministic:
                            dx = -dx
                        else:
                            dx = np.random.randint(-max_s+1, 0)
                            dy = np.random.randint(-max_s+1, max_s)
                    if t < self.n_past or (not self.last_only) or (t == seq_len-1):
                        x[rep, t_, :, sy:sy+32, sx:sx+32] += digit
                        loc[rep, t_, n, 0] = sx+16
                        loc[rep, t_, n, 1] = sy+16
                        t_ += 1
                    sy += dy
                    sx += dx

        x[x>1] = 1.
        return x, y, (loc - 16) / 48.
