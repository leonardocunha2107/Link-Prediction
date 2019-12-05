import networkx as nx
import numpy as np
import random
from sklearn.decomposition import PCA

def draw(embeds,num_nodes=None):
    G = nx.Graph()
    with open("training.txt", "r") as f:
        for line in f:
            line = line.split()
            if line[2] == '1':
                G.add_edge(line[0], line[1])
    pca=PCA(n_components=2)
    x=np.array(embeds)
    x=pca.fit_transform(x)
    pos = {str(i):x[i] for i in range(x.shape[0])}
    n=x.shape[0]
    if num_nodes:
        node_list=[str(i) for i in range(x.shape[0])]
        node_list=random.sample(node_list,n-num_nodes)
        for node in node_list:
            G.remove_node(node)
    nx.draw(G,pos=pos)
    
