import os
import os.path as path
from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import get_stop_words
import numpy as np
from sklearn.decomposition import TruncatedSVD

num_nodes=33226
def tfidf(filename=None):
    dir=path.realpath(path.dirname(__file__))
    file_list=[str(i)+'.txt' for i in range(num_nodes)]
    file_list=[path.join(dir,'text',w) for w in os.listdir(path.join(dir,'text'))]
    sw_set=set(get_stop_words('fr')+get_stop_words('en'))
    
    vectorizer=TfidfVectorizer(input='filename',decode_error='ignore',
                               min_df=3/num_nodes,
                               max_df=0.05,
                               max_features=10000,
                               stop_words=sw_set)
    
    X=vectorizer.fit_transform(file_list)
    decompositor=TruncatedSVD(n_components=500)
    embeds=decompositor.fit_transform(X)
    if filename and path.exists(path.dir(filename)):
        np.save(embeds,filename)
    return embeds,vectorizer

        
                               