import os
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.multiprocessing as mp
from torch.multiprocessing import Process
import random
import time
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP

def sync_initial_weights(model, rank, world_size):
    for param in model.parameters():
        if rank == 0:
            # Rank 0 is sending it's own weight
            # to all it's siblings (1 to world_size)
            for sibling in range(1, world_size):
                dist.send(param.data, dst=sibling)
        else:
            # Siblings must recieve the parameters
            dist.recv(param.data, src=0)

def average_model(model, rank, world_size):
    for param in model.parameters():
         dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
         param.data = param.data/world_size
        
def sync_gradients(model, rank, world_size):
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data = param.grad.data/world_size

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)

def cleanup():
    dist.destroy_process_group()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.il = nn.Linear(1,80)
        self.mi = nn.Linear(80,80)
        self.mi2 = nn.Linear(80,40)
        self.ol = nn.Linear(40,1)
        self.relu = nn.ReLU()
    def forward(self,x):
        hidden1 = self.il(x)
        hissen2 = self.mi(self.relu(hidden1))
        hissen3 = self.mi2(self.relu(hissen2))
        act = self.relu(hissen3)
        out = self.ol(act)
        return out
N=80000
BS = 10000
M = 4 #number of gpu threads
epocs = 20000

x_in = [(20.0/N)*random.randint(0,N) for i in range(N)]
def yfun(i):
    return np.sqrt(i)*np.sin(4*3.14*i/20.0)
y_vals = [yfun(i) for i in x_in]

def mk_minibatch(i, size):  
    s = int(size)
    my_in = x_in[s*i: s*(i+1)]
    my_vals = y_vals[s*i: s*(i+1)]
    return (my_in, my_vals)

batches = [mk_minibatch(i, BS) for i in range(int(N/BS))]
k = len(batches)
batch = []
s = 0
for i in range(M):
    bat = []
    for j in range(int(k/M)):
        bat.append(batches[s+j])
    batch.append(bat)
    s+= int(k/M)

def batchtodev(rank, device):
    btch =batch[rank]
    devb = []
    for x in btch:
        xin = torch.tensor([x[0]], device=device).T
        yin = torch.tensor([x[1]], device=device).T
        devb.append((xin, yin))
    return devb

def run(rank, size):
    setup(rank, size)
    #determine my device IDs.  
    n = torch.cuda.device_count() // size
    device_ids = list(range(rank * n, (rank + 1) * n))
    # create model and move it to device_ids[0]
    device = device_ids[0]
    model = Net().to(device)
   
    sync_initial_weights(model, rank, size)
    btch = batchtodev(rank, device)
    print('batch has ', len(btch), ' elements')
    print("len of btch[0][0] =", len(btch[0][0]))

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005)
    sync_freq =10
    for epoc in range(1, epocs):
        for b in btch:
            optimizer.zero_grad()
            outputs = model(b[0])
            loss = loss_fn(outputs,b[1])
            loss.backward()
            if epoc % sync_freq == 0:
                sync_gradients(model, rank, world_size)
            optimizer.step()
        if epoc % 200 == 0:
            average_model(model,rank, size)
        
        if epoc % 1000 == 0:
            print('epoc %d loss %f'%(epoc, float(loss)))
    if rank == 0:
        torch.save(model.state_dict(), "/tmp/model")

    cleanup()


def launch(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    t0 = time.time()
    launch(run, M)
    print('elapsed = ', time.time()-t0)

    x_in = [0.025*random.randint(0,800) for i in range(800)]
    def yfun(i):
        return np.sqrt(i)*np.sin(4*3.14*i/20.0)
    y_vals = [yfun(i) for i in x_in]
    inputs = torch.tensor([x_in]).T
    targets = torch.tensor([y_vals]).T
    m0 = Net()
    m0.load_state_dict(torch.load("/tmp/model"))
    mouts = m0(inputs)
    loss_fn = nn.MSELoss()
    err = loss_fn(mouts, targets)
    print("err =", err)
