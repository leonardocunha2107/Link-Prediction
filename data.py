import torch
import numpy as np

def get_edge_tensor(device=torch.device("cuda")):
    edges=[]
    with open("training.txt", "r") as f:
        for line in f:
            line = line.split()
            edge=[int(line[0]),int(line[1])]
            if line[2] == '1':
                edges.append(edge)
            
    edges=np.array(edges)
    edge1,edge2=np.hstack([edges[:,0],edges[:,1]]),np.hstack([edges[:,1],edges[:,0]])
    edges_tensor=torch.tensor(np.vstack([edge1,edge2]),device=device)
    return edges_tensor

class  TrainDS:
    def __init__(self,embed):
        self.embed=embed
        self.edges=[]
        self.y
        with open('training.txt','r') as fd:
            for l in fd:
                l=[int(x) for x in l.split()]
                self.edges.append((l[0],l[1]))
                self.y.append(l[2])
    def __len__(self):
        return len(self.y)
    def __getitem__(self,idx):
        edge=self.edges[idx]
        x=torch.cat([self.embed[edge[0]],self.embed[edge[1]]])
        return x,torch.tensor(self.y[idx])
    
class TestDS:    
    def __init__(self,embed):
        self.embed=embed
        self.edges=[]
        with open('testing.txt','r') as fd:
            for l in fd:
                l=[int(x) for x in l.split()]
                self.edges.append((l[0],l[1]))
    def __len__(self):
        return len(self.edges)
    def __getitem__(self,idx):
        edge=self.edges[idx]
        x=torch.cat([self.embed[edge[0]],self.embed[edge[1]]])
        return x