from jssp import JSSP
import numpy as np
from operator import itemgetter
from copy import deepcopy

def tabu_search(jssp,initial_solution,max_iterations=10000):
    n = jssp.n
    m = jssp.m
    n_ops = jssp.n_ops
    lmin = np.ceil(10 + n/m)
    L=1.4*lmin
    it = 0
    tabu_dict = {}
    for i in range(n_ops):
        for j in range(i+1,n_ops):
            tabu_dict[(i,j)] = -L-1
    s_current = deepcopy(initial_solution)
    f_current = jssp.evaluate(initial_solution)
    s_best = deepcopy(s_current)
    f_best = f_current
    while True:
        nbs,swps = jssp.get_neighbours(s_current)
        evaluated = [(i,jssp.evaluate(nb)) for i,nb in enumerate(nbs)]
        costs = sorted(evaluated,key =lambda x:x[1] if x[1] is not None else np.Inf)
        idx = 0
        for c in costs:
            idx = c[0]
            f_new = c[1]
            s =swps[idx]
            swap = (min(s),max(s))
            if it-tabu_dict[swap]> L or f_new < f_best:
                tabu_dict[swap] = it
                break
        f_current = evaluated[idx][1]
        s_current = deepcopy(nbs[idx])
        #print(f_current)

        if f_current < f_best:
            s_best = deepcopy(f_best)
            f_best = f_current
            print(f"new best at {it}-iter: {f_best}")

        it +=1
        if it == max_iterations:
            break


    return s_best,f_best


if __name__ == "__main__":

    # shape: n x ...  
    seq = np.array([[0, 1, 2,4,3],
                    [0, 2, 1,4,3],
                    [1, 0, 2,3,4]])
    
    # shape: m * n if a job is not a machine set it to -1
    proc = np.array([[3, 3, 5],
                     [2, 1, 2],
                     [5, 5, 3],
                     [8, 1, 10],
                     [8, 1, 10]])

    jssp = JSSP(seq,proc)
    initial_solution = jssp.get_feasible()
    tabu_search(jssp,initial_solution)
    #jssp.draw()