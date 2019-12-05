import numpy as np
from pprint import pprint

def stats(p,y):
    """
        p:np.array(int)
        y:np.array(int)
    """
    tp=(p*y).sum()
    fp=((p)*(1-y)).sum()
    tn=((1-p)*(1-y)).sum()
    fn=((1-p)*y).sum()
    
    n=tp+fp+tn+fn
    raw={'n':n,'tp':tp,'fp':fp,'tn':tn,'fn':fn}
    pr,prec=tp/(tp+fn),tp/(tp+fp)
    metrics={'acc':(tp+tn)/n,'pr':pr,'prec':prec,'nr':tn/(tn+fp),'f1':(2*prec*pr)/(prec+pr)}  
    
    pprint(metrics)
    return metrics