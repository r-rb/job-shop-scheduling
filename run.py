from tabu import tabu_search
from jssp import JSSP
import numpy as np
def read_in(file):
    with open(file) as f:
        x = f.readlines()
    
    param = [int(i) for i in x[0].split()]
    n,m = param[0],param[1]
    seq = []
    proc = np.ones((m,n))*-100

    for j in range(n):
        seq.append([])
        data = x[j+1].split()
        for i in range(m):
            seq[j].append(int(data[2*i]))
            proc[int(data[2*i]),j] = int(data[2*i+1])
    seq = np.array(seq)
    return seq,proc



if __name__ == "__main__":
    seq,proc = read_in("medium.text")
    jssp = JSSP(seq,proc)
    initial = jssp.get_feasible()
    ans = tabu_search(jssp,initial)
    print(ans)
