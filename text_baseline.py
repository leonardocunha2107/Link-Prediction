import json
from tqdm import tqdm
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
from utils import stats
def count(ls):
    dic={}
    for w in ls:
        if w in dic:
            dic[w]+=1
        else:
            dic[w]=1
    return dic
def kernel(s1,s2):
    sum=0
    if s1:
        for w in s1:
            if w in s2:
                sum+=min(s1[w],s2[w])
    return sum
with open('node_information/data.json','r') as fd:
    data=json.load(fd)

for idx in data:
    data[idx]=count(data[idx])
    
docs=list(data.keys())
n=len(docs)
dist=[]

def test():
    y=[]
    preds=[]
    with open("training.txt", "r") as f:
        for line in f:
            line = line.split()
            y.append(1 if line[2]=='1' else 0)
            preds.append(kernel(data[line[0]],data[line[1]]))
    return np.array(preds),np.array(y)


p,y=test()
_=stats(p,y)
    