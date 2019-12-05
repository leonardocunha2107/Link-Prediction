import os
import os.path as path
import json
from tqdm import tqdm
import spacy
from stop_words import get_stop_words

class TextDS:
    def __init__(self,force_redata=True):
        self.dir=path.realpath(path.dirname(__file__))+'/'
        if (not path.exists(self.dir+'data.json')) or force_redata:
            if True:#not path.exists(self.dir+'word_list.json') or not path.exists(self.dir+'count_dic.json'):
                print("hey")
                self.create_counts()
            with open(self.dir+'count_dic.json','r') as fd:
                self.count_dic=json.load(fd)
            with open(self.dir+'word_list.json','r') as fd:
                self.word_list=json.load(fd)
                self.trim_word_list()
            
            
        print(f'Vocab Size: {len(self.word_list)}')
        self.vocab={v:i for i,v in enumerate(self.word_list)}
        self.read_data()
        with open(self.dir+'data.json','r') as fd:
            self.data=json.load(fd)
            
    def trim_word_list(self):
        
        #self.word_list=self.word_list[:200000]
        #print(f'Best_count {self.count_dic[self.word_list[-1]]}')
        #aux={}
        assert(len(self.word_list)==len(self.count_dic))
        self.word_list=set(self.word_list)
        
    
    def read_data(self):
        data={}
        for file in tqdm(os.listdir(self.dir+'text/')):
            idx=int(file.split('.')[0])
            data[idx]=[]
            with open(self.dir+'text/'+file,'r',errors="ignore") as fd:
                ls=fd.read()
            ls=ls.split('\n')
            for l in ls:
                l=l.split()
                for w in l:
                    if w in self.word_list:
                        data[idx].append(self.vocab[w])
        with open(self.dir+'data.json','w+') as fd:
            json.dump(data,fd)
            
                
    def create_counts(self):
        count_dic={}
        for file in os.listdir(self.dir+'text/'):
            with open(self.dir+'text/'+file,'r',errors="ignore") as fd:
                aux_set=set()
                for l in fd:
                    l=l.split()
                    
                    for w in l:
                        if len(w)>2:
                            aux_set.add(w)
                for w in aux_set:
                    if w not in count_dic:
                        count_dic[w]=1
                    else:
                        count_dic[w]+=1       
        
        words=list(count_dic)
                
        words.sort(key = lambda  w : count_dic[w])
        count=0
        res=[]
        for w in words:
            if count_dic[w]>1 and count_dic[w]<4:
                res.append(w)
                count+=1
            else:
                count_dic.pop(w)

        with open('count_dic.json','w+') as fd:
            json.dump(count_dic,fd)
        with open('word_list.json','w+') as fd:
            json.dump(res,fd)

if __name__=='__main__':
    print(get_stop_words('fr'))
