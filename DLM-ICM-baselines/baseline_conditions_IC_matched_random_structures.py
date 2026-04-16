# Using NetworkX package and conllu package
# This is baseline conditions module; more conditions can be added as functions
import os
from io import open
import networkx as nx
from operator import itemgetter
import random
from Measures_rand import *
from Measures import *
import treegen as gen
import depgraph as dep

class Random_base(object):
    def __init__(self, tree):                   # tree has an abstract node=0 and real nodes =1,2,... 
        self.tree=tree                          # tree encodes the nodes and edges content in dictionary format. It uses directed graph (DiGraph) feature of networkX package. For example, nodes are encoded like this - tree.nodes={1:{form:'this',POS:'PRN'},2:{...}}   
        self.ls_rand=[]
    
    def num_cross_rand(self,randtree,abs_root):               # requires random tree and its abstract root=10000
        comput=Compute_measures_rand(randtree,abs_root)
        ncross_random=0
        for edgex in randtree.edges:
            if not edgex[0]==abs_root: 
                if comput.is_projective(edgex):                   # checks if edge is projective or not
                    ncross_random += 0
                else:
                    ncross_random += 1
        return ncross_random                                  # returns number of crossings in the random tree 

    def is_equal_num_crossings(self,randtree,abs_root,num_cross_real):    # requires random tree, its abstract root and nCrossings in the real trees
        flag=False
        num_cross_random=self.num_cross_rand(randtree,abs_root)
        if num_cross_random==num_cross_real:                            # checks if no. of crossings matches in real and random tree
            flag=True
        return flag
    
    def is_similar_HD_distribution(self,randtree,abs_root):
        find=Compute_measures_rand(randtree,abs_root)
        rand_tree=nx.DiGraph()
        for edgex in randtree.edges:
            if not edgex[0]==abs_root:
                rand_tree.add_edge(edgex[0],edgex[1])   # regenrates a dummy random tree by removing its abstract root
        
        random_hd_sample=[]
        for edgey in rand_tree.edges:
            random_hd_sample.append(find.dependency_depth(edgey))
        random_hd_sample.sort()
        
        get=Compute_measures(self.tree)
        real_tree = nx.DiGraph()
        for edgez in self.tree.edges:
            if not edgez[0]==0:
                real_tree.add_edge(edgez[0],edgez[1])   # regenrates a dummy real tree by removing its abstract root
        real_hd_sample=[]
        for edgev in real_tree.edges:
            real_hd_sample.append(get.dependency_depth(edgev))
        real_hd_sample.sort()
        return random_hd_sample==real_hd_sample

    def rand_tree(self,n,num_cross_real):                      # requires length and no. of crossings of real tree                
        code=gen.random_pruefer_code(n)                        
        #print("--------------\t"+str(code))
        all_rand_trees = list(gen.directed_trees(gen.tree_from_pruefer_code(code)))  # generates a list of  random tree with Pruefer's code
        random.shuffle(all_rand_trees)                                               
        for treex in all_rand_trees:    
            real_root=next(nx.topological_sort(treex))                            # finds the root of the random tree
            abstract_root=1000                            
            treex.add_edge(abstract_root,real_root)                               # adds an abstract root to the random tree
            for edgex in treex.edges:
                treex.nodes[edgex[1]]['head']=edgex[0]
            if self.is_equal_num_crossings(treex,abstract_root,num_cross_real):   # checks if no. of crossings are equal in real and random tree 
                if self.is_similar_HD_distribution(treex,abstract_root):
                    self.ls_rand.append(treex)                                        # adds the random tree to the list of random trees  
                    break

    def gen_random(self,num_cross_real):                      # requires number of crossings in the real tree
        n = len(self.tree.edges)
        rand_out=[]
        x=0
        if n<16:                                                                 
            while (len(self.ls_rand)==0) and x<3000:                                        # checks if list of random trees is empty        
                x=x+1
                self.rand_tree(n,num_cross_real)                        
                rand_out=self.ls_rand
        return rand_out                                             # returns the list of randomly generated trees
