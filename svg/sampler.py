import os
import torch


class Sampler:
    def __init__(self, device, batch_size=100):
        self.batch_size = batch_size
        self.model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model.pth')
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
