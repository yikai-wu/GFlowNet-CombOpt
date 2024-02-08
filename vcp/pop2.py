import os, sys, time
from itertools import combinations
import networkx as nx
import random
from gurobipy import *

def color(G, PRINT=0, TLIM=600, solution=None):
    """
    ILP:
    ====
    This is hybrid model POP2 from [1]:

    [1] A.Jabrayilov, P.Mutzel "New Integer Linear Programming Models for the Vertex Coloring Problem"

    CODE VERSION:
    =============
    vcp_pop_3_2.py

    PRINT is 3 Bit Number x2x1x0 \in { 0,..,7 }
    x0 :  Gurobi Output    # 1 
    x1 :  layout channel   # 2
    x2 :  debug channel 1  # 4 
    x3 :  debug channel 2  # 8 
    """
    PRINT = 7
    c00 = time.perf_counter()

    for u, v in G.edges():
        if u == v:  
            G.remove_edge(u, v)

    V_0 = G.nodes()
    E_0 = G.edges()

    G, dominator = coloring_preprocessing_fast(G)
    V = G.nodes()
    E = [(u, v) for u, v in G.edges() if u != v]

    if PRINT & 4:
        print('Reduction (V,E): (%s,%s) --> (%s,%s)    %0.2F sec' % (len(V_0), len(E_0), len(V), len(E), (time.perf_counter() - c00)))

    ub = len(set(nx.greedy_color(G).values()))
    H = ub
    c0 = time.perf_counter()
    Qq = best_clique(G, H, PRINT)
    if PRINT & 4:
        print('Time Qq:', (time.perf_counter() - c0))
    c0 = time.perf_counter()

    lb = len(Qq)

    if PRINT & 4:
        print('(V,E,lb,ub,H) = (%s,%s,%s,%s,%s)' % (len(V), len(E), lb, ub, H))

    if lb == ub:
        rt = '%.2F' % (time.perf_counter() - c00)
        return len(V_0), len(E_0), len(V), len(E), len(Qq), H, lb, ub, 'init', rt

    p = Model("coloring")
    p.params.OutputFlag = 1 & PRINT
    p.params.SEED = 1  # randomized=False
    p.params.THREADS = 1  # nur ein thread
    p.params.TimeLimit = TLIM

    # POP variables
    y = {(i, u): p.addVar(vtype=GRB.BINARY) for u in V for i in range(H-1)}
    for u in V:
        y[H-1, u] = p.addVar(0, 0)
    # ASS
    x = {(u, i): p.addVar(vtype=GRB.BINARY) for u in V for i in range(H)}

    Q = Qq[:-1]
    q = Qq[-1]

    if PRINT & 8:
        print('Q:', Q)
    if PRINT & 8:
        print('q:', q)

    # POP
    for i in range(len(Q)):
        y[i, Q[i]] = p.addVar(0, 0)
    for i in range(1, len(Q)):
        y[i-1, Q[i]] = p.addVar(1, 1)
    y[len(Q)-1, q] = p.addVar(1, 1)
    # ASS
    for i in range(len(Q)):
        x[Q[i], i] = p.addVar(1, 1)

    p.update()

    p.setObjective(1 + sum([y[i, q] for i in range(H)]), GRB.MINIMIZE)

    # Constraints
    for u in V:
        for i in range(H-1):
            p.addConstr(y[i, u] - y[i+1, u] >= 0)

    for u in V:
        for i in range(H):
            p.addConstr(y[i, q] - y[i, u] >= 0)  # (*1)

    # POP -to-> AP
    for u in V:
        p.addConstr(x[u, 0] == 1 - y[0, u])  # (1)
        for i in range(1, H):
            p.addConstr(x[u, i] == y[i-1, u] - y[i, u])  # (1)

    # ( Partial Ordering with AP )
    for u, v in E:
        for i in range(H):
            p.addConstr(x[u, i] + x[v, i] <= 1)  # (*3)

    if PRINT & 4:
        print('Time write IP:', (time.perf_counter() - c0))

    p.optimize()

    rtIP = '%.2F' % p.runtime
    rt = '%.2F' % (time.perf_counter() - c00)

    print_solution_to_check_sol_file(dominator, H, y, solution)
    if PRINT & 2:
        print_solution(dominator, H, y, Qq)
    return len(V_0), len(E_0), len(V), len(E), lb, ub, p.objBound, p.objVal, rtIP, rt

def best_clique(G, H, PRINT):
    t0 = time.perf_counter()
    lenV = len(G.nodes())
    E = {(u, v) for u, v in G.edges() if u != v}
    number_vars = H * lenV
    bestQplusCutQ = 0
    Q = []
    max_try = int(300 * len(E) / lenV)
    Gcomplement = nx.complement(G)
    for i in range(max_try):
        random.seed(i)
        _Q = nx.maximal_independent_set(Gcomplement)
        setQ = set(_Q)
        lenQ = len(_Q)
        if lenQ == H:
            return _Q
        cutQ = sum([1 for u, v in E if len(setQ & {u, v}) == 1])
        QplusCutQ = H * lenQ + cutQ
        if bestQplusCutQ < QplusCutQ:
            bestQplusCutQ = QplusCutQ
            Q = _Q
            remaining_number_vars = number_vars - QplusCutQ
            string = '\t|Q|: %s   #Vars: %s/%s     %s sec\r' % (lenQ, remaining_number_vars, number_vars, time.perf_counter() - t0)
            if PRINT & 8:
                sys.stderr.write(string)
                sys.stdout.flush()
        if time.perf_counter() - t0 > 60:
            break
    if PRINT & 4:
        sys.stdout.write(string)
        sys.stdout.flush()
        print()
    return Q

def get_solution(dominator, H, y):
    epsilon = 0.000000001
    sol = {}
    for v in dominator:
        u = dominator[v]
        while u != dominator[u]:  
            u = dominator[u]
        if abs(y[0, u].x) <= epsilon:
            sol[u] = 1
        for i in range(1, H):
            if abs(y[i, u].x + (1 - y[i - 1, u].x)) <= epsilon:
                sol[u] = i + 1
        sol[v] = sol[u]
    return sol

def print_solution(dominator, H, y, Q):
    try:
        sol = get_solution(dominator, H, y)
        bold = '\033[1;31m'
        norm = '\033[0m'
        for c in sorted(set(sol.values())):
            print(f"{c} ...", end=" ")
            for i in range(1, len(sol) + 1):
                v = str(i)  # we assume: V={ '1', '2', ... '|V|' }
                if sol[v] == c:
                    if v in Q:
                        print(f'{bold}{v}{norm}', end=" ")
                    else:
                        print(v, end=" ")
            print()
        print()
    except KeyError:
        print("\nKeyError: print_solution works only if V={ '1', '2', ... '|V|' }")

def print_solution_to_check_sol_file(dominator, H, y, check_sol_file):
    if check_sol_file is not None:
        sol = get_solution(dominator, H, y)
        os.system('echo -n > %s' % check_sol_file)
        for i in range(1, len(sol) + 1):
            v = str(i)  # we assume: V={ '1', '2', ... '|V|' }
            os.system('echo %s >> %s' % (sol[v], check_sol_file))

def coloring_preprocessing_fast(G):
    dominator = {v: v for v in G.nodes()}
    n = len(G.nodes()) + 1
    while len(G.nodes()) < n:
        n = len(G.nodes())
        adj = {v: set(G.neighbors(v)) for v in G}
        Vredundant = []
        for u, v in combinations(G.nodes(), 2):
            if adj[u] <= adj[v]:
                Vredundant.append(u)
                dominator[u] = v
            elif adj[v] <= adj[u]:
                Vredundant.append(v)
                dominator[v] = u
        G.remove_nodes_from(Vredundant)
    return G, dominator
