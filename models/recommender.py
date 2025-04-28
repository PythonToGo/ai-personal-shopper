import faiss
import numpy as np
import os
import pickle

class FaissRecommender:
    def __init__(self, dim=512, index_path="data/faiss_index.pkl"):
        self.index_path = index_path
        self.dim = dim
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            with open(index_path.replace('.faiss', '.pkl'), 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(dim)
            self.metadata = []
    
    def add(self, embedding, meta):
        """ add new embedding"""
        self.index.add(np.expand_dims(embedding, axis=0))
        self.metadata.append(meta)
    
    def search(self, query_embedding, topk=5):
        """ search topk nearest query embedding"""
        D, I = self.index.search(np.expand_dims(query_embedding, axis=0), topk)
        return [(self.metadata[i], float(D[0][idx])) for idx, i in enumerate(I[0])]
    
    def save(self):
        """ save index and metadata"""
        faiss.write_index(self.index, self.index_path)
        with open(self.index_path.replace('.faiss', '.pkl'), 'wb') as f:
            pickle.dump(self.metadata, f)
