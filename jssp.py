import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
import pygraphviz as pgv

class JSSP:
    def __init__(self,seq,proc):
        self.n = len(seq)
        self.n_ops = len(seq.flatten())
        self.m = len(proc)
        self.proc = proc
        self.seq = seq
        self.V = tuple(list(range(-1,self.n_ops+1))) #node (operations)
        self.A = [] # conjunctive arcs
        self.E = [] # disjunctive arcs

        self.init_dicts()

        u = 0
        for i,job in enumerate(seq):
            for j,mach in enumerate(job):
                self.MJ[u] = (mach,i)
                if j > 0 :
                    self.JP[u] = u-1
                    self.A.append((u-1,u))
                if j+1<len(job):
                    self.JS[u] = u+1
                    self.A.append((u,u+1))
                if j==0:
                    self.JP[u] = -1
                    self.A.append((-1,u))
                if j==len(job)-1:
                    self.JS[u]= self.n_ops
                    self.A.append((u,self.n_ops))
                u+=1

        # INEFFCIENT
        for u in self.V:
            for v in self.V:
                mu,ju = self.getMJ(u)
                mv,jv = self.getMJ(v)
                if mu == mv and ju != jv:
                    self.E.extend([(u,v),(v,u)])
        
        self.ops = {v:k for (k,v) in self.MJ.items()}

        print(self.ops)

        s = self.feasible_solution()

        self.draw_solution(s)

    def draw(self):
        G = nx.DiGraph()
        G.add_nodes_from([i+1 for i in self.V])
        G.add_edges_from([(i[0]+1,i[1]+1) for i in self.A],style ='solid')
        G.add_edges_from([(i[0]+1,i[1]+1) for i in self.E],style ='dashed')
        # set defaults
        G.graph['graph']={'rankdir':'LR'}
        G.graph['node']={'shape':'circle'}
        G.graph['edges']={'arrowsize':'2.0'}
        A = to_agraph(G)
        #print(A)
        A.layout('dot')
        A.draw('graph.png')
    
    def draw_solution(self,disjunctive_edges):
        G = nx.DiGraph()
        G.add_nodes_from([i+1 for i in self.V])
        G.add_edges_from([(i[0]+1,i[1]+1) for i in self.A],style ='solid')
        G.add_edges_from([(i[0]+1,i[1]+1) for i in disjunctive_edges],style ='dashed')
        # set defaults
        G.graph['graph']={'rankdir':'LR'}
        G.graph['node']={'shape':'circle'}
        G.graph['edges']={'arrowsize':'2.0'}
        print(nx.is_directed_acyclic_graph(G))
        A = to_agraph(G)
        #print(A)
        A.layout('dot')
        A.draw('solution.png')

    def init_dicts(self):
        self.MJ = {u:(None,None) for u in self.V}
        self.JS = {u:None for u in self.V}
        self.JP = {u:None for u in self.V}
        self.MS = dict()
        self.MP = dict()
    
    def feasible_solution(self):
        # need to add
        disjunctive_edges = []
        mach_schedule = [[] for i in range(self.m)]

        for job,order  in enumerate(self.seq):
            for machine in order:
                mach_schedule[machine].append(self.ops[(machine,job)])
        print(mach_schedule)
        
        for machine in range(self.m):
            njobs = len(mach_schedule[machine])
            schedule = mach_schedule[machine]
            for x in range(njobs):
                if x < njobs-1:
                    print((schedule[x],schedule[x+1]))
                    disjunctive_edges.append((schedule[x],schedule[x+1])) 
        print(disjunctive_edges)
        return disjunctive_edges


    def getJS(self,u):
        return self.JS[u]
    def getJP(self,u):
        return self.JS[u]
    def getMS(self,u):
        return self.MS[u]
    def getMP(self,u):
        return self.MP[u]
    def getMJ(self,u):
        return self.MJ[u]

if __name__ == "__main__":

    # shape: n x ...  
    seq = np.array([[0, 1, 2],
                    [0, 2, 1],
                    [1, 0, 2]])
    
    # shape: m * n if a job is not a machine set it to -1
    proc = np.array([[3, 2, 5],
                     [3, 1, 5],
                     [5, 2, 3]])

    jssp = JSSP(seq,proc)

    #jssp.draw()