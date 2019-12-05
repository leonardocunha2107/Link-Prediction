import numpy as np
from pprint import pprint
import torch

def stats(p,y):
    """
        p:np.array(int)
        y:np.array(int)
    """
    tp=(p*y).sum().data
    fp=((p)*(1-y)).sum().data
    tn=((1-p)*(1-y)).sum().data
    fn=((1-p)*y).sum().data
    
    n=tp+fp+tn+fn
    raw={'n':n,'tp':tp,'fp':fp,'tn':tn,'fn':fn}
    pr,prec=tp/(tp+fn),tp/(tp+fp)
    metrics={'acc':(tp+tn)/n,'pr':pr,'prec':prec,'nr':tn/(tn+fp),'f1':(2*prec*pr)/(prec+pr)}  
    
    pprint(metrics)
    return metrics

def loader_stats(model,embedder,dataloader,threshold=0.5):
    tp,fp,tn,fn=0,0,0,0
    with torch.no_grad():
        model.eval()  # set model to evaluation mode
        
        for x,y in dataloader:
            x = x.to(device=device, dtype=torch.long)  
            y = y.to(device=device, dtype=torch.long)

            x=embedder(x)

            scores = model(x)
            preds = (scores[:,1]>threshold).long()
            comp=(preds==y).long()

            ##calcule standard stats
            tp+=(comp*y).sum().data
            fp+=((1-comp)*(1-y)).sum().data
            tn+=((comp)*(1-y)).sum().data
            fn+=((1-comp)*y).sum().data

            
    tp,fp,tn,fn=float(tp)+1e-5,float(fp)+1e-5,float(tn)+1e-5,float(fn)+1e-5
    n=tp+fp+tn+fn
    raw={'n':n,'tp':tp,'fp':fp,'tn':tn,'fn':fn}

    pr,prec=tp/(tp+fn),tp/(tp+fp)
    metrics={'acc':(tp+tn)/n,'pr':pr,'prec':prec,'nr':tn/(tn+fp),'f1':(2*prec*pr)/(prec+pr)}  

    log={'raw':raw,'metrics':metrics}
    pprint(log['metrics'])
    return log

def train_epoch(model,optim,train_dl):
    loss_avg=0
    device=torch.device("cuda")
    model.train()
    for x,y in train_dl:
        optim.zero_grad()
        x = x.to(device=device, dtype=torch.float)  
        y = y.to(device=device, dtype=torch.long)


        scores = model(x)
        
        loss = loss_fn(scores, y)
        loss_avg+=loss.data
        loss.backward(retain_graph=False)

        optim.step()

    
    print('Loss %.4f' %loss)
    
    return loss_avg