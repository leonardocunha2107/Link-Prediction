from sklearn.linear_model import LogisticRegression
from utils import stats
import numpy as np
d=int(embeds.shape[1]*0.2)
embeds=embeds[:,:d]
lr=LogisticRegression()
x=[]
y=[]
with open("training.txt", "r") as f:
    for line in f:
        line = line.split()
        x.append(np.hstack([embeds[int(line[0])],embeds[int(line[1])]]))
        if line[2] == '1':
            y.append(1)
        else:
            y.append(0)
x=np.array(x)
y=np.array(y)
n,d=x.shape
idxs=np.arange(n)
np.random.shuffle(idxs)
id_lim=int(0.8*n)
x_train,y_train=x[idxs[:id_lim]],y[idxs[:id_lim]]
x_val,y_val=x[idxs[id_lim:]],y[idxs[id_lim:]]
lr.fit(x_train,y_train)

print('Train stats')
_=stats(lr.predict(x_train),y_train)

print('Val stats')
_=stats(lr.predict(x_val),y_val)