import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
from networkx.algorithms import dag_longest_path,dag_longest_path_length
import pygraphviz as pgv
import pprint as pp
from copy import deepcopy

class JSSP:
    def __init__(self,seq,proc):
        self.n = len(seq)
        self.n_ops = len(seq.flatten())
        self.m = len(proc)
        self.proc = proc
        self.seq = seq
        self.V = tuple(list(range(-1,self.n_ops+1))) #node (operations)
        self.A = [] # conjunctive arcs
        self.E = [] # disjunctive arcs\
        self.G = None

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
        self.processing_time = [proc[self.MJ[i]] if i>-1 and i < self.n_ops else 0 for i in range(-1,self.n_ops+1)]
        print(self.processing_time)
        # INEFFCIENT - might not actually need this 
        for u in self.V:
            for v in self.V:
                mu,ju = self.MJ[u]
                mv,jv = self.MJ[v]
                if mu == mv and ju != jv:
                    self.E.extend([(u,v),(v,u)])
        
        self.ops = {v:k for (k,v) in self.MJ.items()}

        print(self.ops)

        s = self.feasible_solution()

        nbs = self.return_neighbours(s)
        #pp.pprint(nbs)

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
        print(self.is_feasible(disjunctive_edges))
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

    def is_feasible(self,E):
        self.G = nx.DiGraph()
        self.G.add_nodes_from([i for i in self.V])
        self.G.add_weighted_edges_from([(i[0],i[1],self.processing_time[i[0]+1]) for i in self.A])
        self.G.add_weighted_edges_from([(i[0],i[1],self.processing_time[i[0]+1]) for i in E])
        return nx.is_directed_acyclic_graph(self.G)
    
    def is_feasible_sol(self,E):
        G = nx.DiGraph()
        G.add_nodes_from([i for i in self.V])
        G.add_weighted_edges_from([(i[0],i[1],self.processing_time[i[0]+1]) for i in self.A])
        G.add_weighted_edges_from([(i[0],i[1],self.processing_time[i[0]+1]) for i in E])
        return nx.is_directed_acyclic_graph(G)
    
    def cost(self,E):
        if self.is_feasible(E):
            #print(dag_longest_path(self.G))
            #print(dag_longest_path_length(self.G))
            return dag_longest_path(self.G) 

    def get_blocks(self,path):
        blocks = []
        machine_tracker = -1
        for p in path:
            if p != -1:
                mch,_ = self.MJ[p]
                if mch != machine_tracker:
                    machine_tracker = mch
                    blocks.append([p])
                elif mch == machine_tracker:
                    blocks[-1].append(p)
        return blocks
                
    def return_neighbours(self,E):
        path = self.cost(E)
        blocks = self.get_blocks(path)
        nbs = []
        for block in blocks:
            n_block = len(block)
            if n_block<=1:
                continue
            for i in range(n_block):
                for j in range(i+1,n_block):
                    u = block[i]
                    v = block[j]
                    
                    js = self.JS[v]
                    if js in path:
                        nbs.append(self.insert_after(u,v,E))
            f = 0 # placeholder for first element
            for j,v in enumerate(block):
                if j==0:
                    f = v
                    continue
                nbs.append(self.move_to_start(f,v,E))

            for i in range(n_block):
                for j in range(i+1,n_block):
                    u = block[i]
                    v = block[j]
                    
                    jp = self.JP[u]
                    if jp in path:
                        nbs.append(self.insert_after(u,v,E))
            
            for i,u in enumerate(block):
                jp = self.JP[u]
                print(f" IM u : {u}")
                if jp in path:
                    nbs.append(self.move_succesive(u,E))

        return nbs
    
    def insert_after(self,u,v,E):
        E_new = deepcopy(E) # make sure this copies by value not ref
        
        # original edge:
        ms_u,mp_u,ms_v,mp_v = None,None,None,None
        for edge in E:
            if edge[0] == u:
                ms_u = edge[1]
                E_new.remove(edge)
            if edge[1] == u:
                mp_u = edge[0]
                E_new.remove(edge)
            if edge[0] == v:
                ms_v = edge[1]
                E_new.remove(edge)
            if edge[1] == v:
                mp_v = edge[0]
        
        E_new.append((v,u))
        if ms_v is not None:
            E_new.append((u,ms_v))
        if mp_u is not None:
            E_new.append((mp_u,ms_u))
        #print(f"{u},{v}, is feasible? :{self.is_feasible_sol(E_new)}")
        return E_new
    
    def move_to_start(self,f,v,E):
        E_new = deepcopy(E) # moving v to start
        # original edge:
        ms_f,mp_f,ms_v,mp_v = None,None,None,None

        for edge in E:
            if edge[0] == f:
                ms_f = edge[1]
            if edge[0] == v:
                ms_v = edge[1]
                E_new.remove(edge)
            if edge[1] == v:
                mp_v = edge[0]
                E_new.remove(edge)

        E_new.append((v,f))
        if ms_v is not None and mp_v is not None:
            E_new.append((mp_v,ms_v))

        #print(f"{f},{v}, is feasible? :{self.is_feasible_sol(E_new)}")
        return E_new
    
    def insert_before(self,u,v,E):
        E_new = deepcopy(E) # make sure this copies by value not ref
        
        ms_u,mp_u,ms_v,mp_v = None,None,None,None
        for edge in E:
            if edge[0] == u:
                ms_u = edge[1]
            if edge[1] == u:
                mp_u = edge[0]
                E_new.remove(edge)
            if edge[0] == v:
                ms_v = edge[1]
                E_new.remove(edge)
            if edge[1] == v:
                mp_v = edge[0]
                E_new.remove(edge)

        E_new.append((v,u))
        if ms_v is not None and mp_v is not None:
            E_new.append((mp_v,ms_v))
        if mp_u is not None:
            E_new.append((mp_u,v))
        
        print(f"{u},{v}, is feasible? :{self.is_feasible_sol(E_new)}")
        return E_new
        
    def move_succesive(self,u,E):
        E_new = deepcopy(E) # make sure this copies by value not ref
        ms_u,mp_u = None,None
        for edge in E:
            if edge[0] == u:
                ms_u = edge[1]
                E_new.remove(edge)
            if edge[1] == u:
                mp_u = edge[0]
                E_new.remove(edge)
        msms_u = None
        for edge in E:
            if edge[0] == ms_u:
                msms_u = edge[1]
                E_new.remove(edge)
        if ms_u is None:
            return deepcopy(E)
        if mp_u is not None:
            E_new.append((mp_u,ms_u))
        if msms_u is not None:
            E_new.append((u,msms_u))
        if ms_u is not None:
            E_new.append((ms_u,u))
        return E_new
        

if __name__ == "__main__":

    # shape: n x ...  
    seq = np.array([[0, 1, 2],
                    [0, 2, 1],
                    [1, 0, 2]])
    
    # shape: m * n if a job is not a machine set it to -1
    proc = np.array([[3, 3, 5],
                     [2, 1, 2],
                     [5, 5, 3]])

    jssp = JSSP(seq,proc)

    #jssp.draw()