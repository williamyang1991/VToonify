import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import faiss
import collections

OptParams = collections.namedtuple('OptParams', 'lr batch_size epochs ' +
                                                'decay_epochs decay_rate ')
OptParams.__new__.__defaults__ = (None, None, None, None, None)


class _netT(nn.Module):
    def __init__(self, xn, yn):
        super(_netT, self).__init__()
        self.xn = xn
        self.yn = yn
        self.lin1 = nn.Linear(xn, 128, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.lin2 = nn.Linear(128, 128, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.lin_out = nn.Linear(128, yn, bias=False)
        self.relu = nn.ReLU(True)

    def forward(self, z):
        z = self.lin1(z)
        z = self.bn1(z)
        z = self.relu(z)
        z = self.lin2(z)
        z = self.bn2(z)
        z = self.relu(z)
        z = self.lin_out(z)
        return z

class _netLin(nn.Module):
    def __init__(self, xn, yn):
        super(_netLin, self).__init__()
        self.xn = xn
        self.yn = yn
        self.lin = nn.Linear(xn, yn, bias=True)

    def forward(self, z):
        z = self.lin(z)
        return z


class _ICP():
    def __init__(self, e_dim, z_dim):
        self.e_dim = e_dim
        self.z_dim = z_dim
        self.netT = _netT(e_dim, z_dim)#.cuda()

    def train(self, z_np, opt_params):
        self.opt_params = opt_params
        for epoch in range(opt_params.epochs):
            self.train_epoch(z_np, epoch)

    def train_epoch(self, z_np, epoch):
        # Compute batch size
        batch_size = self.opt_params.batch_size
        n, d = z_np.shape
        batch_n = n // batch_size
        rp = np.random.permutation(n)

        # Compute learning rate
        decay_steps = epoch // self.opt_params.decay_epochs
        lr = self.opt_params.lr * self.opt_params.decay_rate ** decay_steps
        # Initialize optimizers
        optimizerT = optim.Adam(self.netT.parameters(), lr=lr,
                                betas=(0.5, 0.999), weight_decay=1e-5)
        criterion = nn.MSELoss().cuda()
        self.netT.train()

        M = batch_n * 2
        e_np = np.zeros((M * batch_size, self.e_dim))
        Te_np = np.zeros((M * batch_size, self.z_dim))
        for i in range(M):
            e = torch.randn(batch_size, self.e_dim).cuda()
            y_est = self.netT(e)
            e_np[i * batch_size: (i + 1) * batch_size] = e.cpu().data.numpy()
            Te_np[i * batch_size: (i + 1) * batch_size] = y_est.cpu().data.numpy()

        nbrs = faiss.IndexFlatL2(self.z_dim)
        nbrs.add(Te_np.astype('float32'))
        _, indices = nbrs.search(z_np.astype('float32'), 1)
        indices = indices.squeeze(1)


        # Start optimizing
        er = 0

        for i in range(batch_n):
            self.netT.zero_grad()
            # Put numpy data into tensors
            idx_np = i * batch_size + np.arange(batch_size)
            e = torch.from_numpy(e_np[indices[rp[idx_np]]]).float().cuda()
            z_act = torch.from_numpy(z_np[rp[idx_np]]).float().cuda()
            z_est = self.netT(e)
            loss = criterion(z_est, z_act)
            loss.backward()
            er += loss.item()
            optimizerT.step()

        print("Epoch: %d Error: %f" % (epoch, er / batch_n))


class ICPTrainer():
    def __init__(self, f_np, d):
        self.f_np = f_np
        self.icp = _ICP(d, f_np.shape[1])

    def train_icp(self, n_epochs):
        uncca_opt_params = OptParams(lr=1e-3, batch_size=128, epochs=n_epochs,
                                     decay_epochs=50, decay_rate=0.5)
        self.icp.train(self.f_np, uncca_opt_params)