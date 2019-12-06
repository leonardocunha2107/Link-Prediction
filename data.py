import torch
import numpy as np
import random

def edge_tensors(val_cut=0.8,device=torch.device("cuda")):
    neg_edges=[]
    pos_edges=[]
    with open("training.txt", "r") as f:
        for line in f:
            line = line.split()
            edge=[int(line[0]),int(line[1])]
            if line[2] == '1':
                pos_edges.append(edge)
            else: neg_edges.append(edge)
    random.shuffle(pos_edges)
    random.shuffle(neg_edges)
    pos_edges,neg_edges=np.array(pos_edges),np.array(neg_edges)
    postrain_edges,negtrain_edges=pos_edges[:int(len(pos_edges)*val_cut)],neg_edges[:int(len(neg_edges)*val_cut)]
    posval_edges,negval_edges=pos_edges[int(len(pos_edges)*val_cut):],neg_edges[int(len(neg_edges)*val_cut):]
    val_edges=np.vstack([posval_edges,negval_edges])
    
    edge1,edge2=np.hstack([postrain_edges[:,0],postrain_edges[:,1]]),np.hstack([postrain_edges[:,1],postrain_edges[:,0]])
    postrain_edges_tensor=torch.tensor(np.vstack([edge1,edge2]),device=device,dtype=torch.long)
    
    edge1,edge2=np.hstack([negtrain_edges[:,0],negtrain_edges[:,1]]),np.hstack([negtrain_edges[:,1],negtrain_edges[:,0]])
    negtrain_edges_tensor=torch.tensor(np.vstack([edge1,edge2]),device=device,dtype=torch.long)
    
    edge1,edge2=np.hstack([val_edges[:,0],val_edges[:,1]]),np.hstack([val_edges[:,1],val_edges[:,0]])
    valedges_tensor=torch.tensor(np.vstack([edge1,edge2]),device=device,dtype=torch.long)
    
    ytrain=torch.tensor([1]*(postrain_edges_tensor.shape[1])+[0]*(negtrain_edges_tensor.shape[1]))
    yval=torch.tensor(([1]*(len(posval_edges))+[0]*(len(negval_edges)))*2)
    return postrain_edges_tensor,\
            (torch.cat([postrain_edges_tensor,negtrain_edges_tensor],axis=1).T,ytrain),\
            (valedges_tensor.T,yval)


class  TrainDS:
    def __init__(self,embed):
        self.embed=embed
        self.edges=[]
        self.y=[]
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

class XYDS:
    def __init__(self,data,embed):
        self.x,self.y=data
        self.embed=embed
    def __getitem__(self,id):
        edge=self.x[id]
        x=torch.cat([self.embed[edge[0]],self.embed[edge[1]]])
        return x,self.y[id]
    def __len__(self):
        return len(self.x)
    
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

    
def generate_csv(model,embeds,filename='test.csv',thr=0.5):
    predictions = []
    with open("testing.txt", "r") as f:
        for l in f:
            l = [int(x) for x in l.split()]
            x=torch.cat([embeds[l[0]],embeds[l[1]]])
            x=model(x)[1]
            if x>thr:
                predictions.append("1")
            else: 
                predictions.append("0")
            
    
    predictions = zip(range(len(predictions)), predictions)
    # Write the output in the format required by Kaggle
    with open(filename,"w") as pred:
        csv_out = csv.writer(pred)
        csv_out.writerow(['id','predicted'])
        for row in predictions:
            csv_out.writerow(row)