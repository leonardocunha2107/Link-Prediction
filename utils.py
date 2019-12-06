import numpy as np
from pprint import pprint
import torch
import torch.nn as nn


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

def train_epoch(model,optim,train_dl,loss_fn=nn.CrossEntropyLoss(),device=torch.device("cuda")):
    loss_avg=0
    model.train()
    i=0
    for x,y in train_dl:
        optim.zero_grad()
        x = x.to(device=device, dtype=torch.float)  
        y = y.to(device=device, dtype=torch.long)


        scores = model(x)
        
        loss = loss_fn(scores, y)
        loss_avg+=loss.data
        loss.backward(retain_graph=False)

        optim.step()
        i+=1

    
    print('Loss %.4f' %(loss_avg/i))
    
    return loss_avg
def loader_stats(model,dataloader,threshold=0.5,device=torch.device("cuda")):
    tp,fp,tn,fn=0,0,0,0
    with torch.no_grad():
        model.eval()  # set model to evaluation mode
        
        for x,y in dataloader:
            x = x.to(device=device, dtype=torch.float)  
            y = y.to(device=device, dtype=torch.long)


            scores = model(x)
            scores=torch.nn.functional.softmax(scores)
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