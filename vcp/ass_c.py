import os, sys, time
from itertools import combinations
import networkx as nx
import random
from gurobipy import *

def color( G, PRINT=0, TLIM=600, solution=None): #{
    """ 
    ILP:
    ====
	Classic Assignment Formulation and some symmetry breaking constraints from [1],
	which is referenced ass ASS+(c) in [2].
    
	[1] I.Mendez, P.Zabala "A Branch-and-Cut Algorithm for Graph Coloring"
	[2] A.Jabrayilov, P.Mutzel "New Integer Linear Programming Models for the Vertex Coloring Problem"
	
    CODE VERSION:
    =============
	vcp_ass_3.py

    PRINT is 3 Bit Number x2x1x0 \in { 0,..,7 }
	x0 :  Gurobi Output	# 1 
	x1 :  layout chanel   	# 2
	x2 :  debug  chanel 1 	# 4 
	x3 :  debug  chanel 2 	# 8 

    """
    c00 = time.clock()
    
    """ remove reflexive edges (v,v) \in E """
    for u,v in G.edges():
	if u == v:  G.remove_edge(u,v)

    V_0 = G.nodes()
    E_0 = G.edges()

    G,dominator = coloring_preprocessing_fast( G )

    V = G.nodes()
    E = [ (u,v) for u,v in G.edges() if u != v ]

    if PRINT&4: print 'Reduction (V,E): (%s,%s) --> (%s,%s)    %0.2F sec  ' % (
		len(V_0), len(E_0), len(V), len(E), (time.clock() - c00)); 


    ub = len(set(nx.greedy_color(G).values()))
    #  _   _ 
    # | | | |
    # | |_| |
    # |  _  |
    # |_| |_|
    #        
    H = ub

    if PRINT&4:	print '(V,E,H) = (%s,%s,%s)' % (len(V),len(E),ub)
    #   ___  
    #  / _ \ 
    # | | | |
    # | |_| |
    #  \__\_\
    #        
    c0 = time.clock()
    Q = best_clique( G, ub )
    if PRINT&4:	print 'Time Qq:', (time.clock() - c0)

    lb = len(Q)

    if lb == ub: 
	rt   = '%.2F' %(time.clock() - c00)
	return len(V_0), len(E_0), len(V), len(E), len(Qq), H, lb, ub, 'init', rt

    # create model
    p = Model("vcp-ass")
    p.params.OutputFlag = 1&PRINT
    p.params.SEED       = 1	    # randomized=False
    p.params.THREADS    = 1	    # nur ein thread
    p.params.TimeLimit  = TLIM
    
    # create variables
    x = { (u,i): p.addVar(vtype=GRB.BINARY)     for u in V   for i in range(H) }
    w = {  i   : p.addVar(vtype=GRB.BINARY)     for i in range(H) }

    if PRINT&8:	print 'Q:', Q
    for i in range(len(Q)):     x[ Q[i], i ] = p.addVar(1,1)

    p.update()

    """ objective """ 
    p.setObjective(
	sum([ w[i] for i in range(H) ])
	, GRB.MINIMIZE
    )

    """ constraints """
    for u in V:	
	p.addConstr( sum([ x[u,i] for i in range(H) ]) == 1.0 ) # (1)

    for u,v in E:
	for i in range(H):
	    p.addConstr( x[u,i] + x[v,i] <= w[i] ) # (2)

    # [1, Seite 2, CP = SCP \cap ... ]
    for i in range(H): 
	p.addConstr( w[i] <= sum([ x[u,i]  for u in V ]) )

    # (8)
    # [1, Seite 2, CP = SCP \cap ... ]
    for i in range(1,H):
	p.addConstr( w[i-1] >= w[i] )


    p.optimize()


    rtIP = '%.2F' % p.runtime 
    rt   = '%.2F' %(time.clock() - c00)

    print_solution_to_check_sol_file(dominator,H,x,solution)
    return len(V_0), len(E_0), len(V), len(E), lb, ub, p.objBound, p.objVal, rtIP, rt
#} 

def best_clique( G, H  ): #{
    t0 = time.clock()
    lenV = len(G.nodes())
    E = { (u,v) for u,v in G.edges() if u != v }
    Q=[]
    max_try=(300*len(E)/lenV) 
    Gcomplement = nx.complement( G )
    for i in range( max_try ):
	random.seed(i)
	_Q=nx.maximal_independent_set( Gcomplement )
	if len(Q) < len(_Q):
	    Q = _Q
	    string='\t|Q|: %s   %s sec\r' % ( len(Q), time.clock() - t0 )
	    sys.stderr.write( string );sys.stdout.flush()
	if time.clock() - t0 > 60:
	    break

    sys.stdout.write( string );sys.stdout.flush()
    print
    return Q
#}

def print_solution_to_check_sol_file(dominant_of,H,y,check_sol_file): #{
    if check_sol_file != None:
	epsilon=0.000000001
	sol=[-1]*(len(dominant_of)+1)
	for v in dominant_of:

	    u = dominant_of[ v ]
	    while u != dominant_of[ u ]:
		u = dominant_of[ u ]

	    for i in range(0,H):
		if abs( y[u,i].x - 1 ) <= epsilon:
		    sol[ int(u) ] = i+1

	    sol[ int(v) ] = sol[ int(u) ]
	    
	os.system( 'echo -n > ' + check_sol_file )
	for c in sol:
	    if c > -1:
		os.system( 'echo ' + str(c) + ' >> ' + check_sol_file )
#}

def coloring_preprocessing_fast( G ): #{
    """
	Removes dominated nodes.

	WARNING:
	========
	This method are using the condition
		if adj[u] <= adj[v]:	# <=
	instead of
		if adj[u] < adj[v]:	# <
	and this is correct, iff graph has no reflexive edges (u,v) with u == v.
	Hence the reflexive edges must be deleted, before this method begins.  

	EFFICIENCE:
	========== 
	This implementation is faster than coloring_preprocessing_old:

	coloring_preprocessing_old  Time(ash958GPIA.col) = 19.489439 sec
	coloring_preprocessing_fast Time(ash958GPIA.col) = 0.537492  sec 
    """ 
    dominator = { v: v for v in G.nodes() }
    n=len(G.nodes())+1
    while len(G.nodes()) < n:
	n=len(G.nodes())
	adj={ v:set(G.neighbors(v)) for v in G }
	Vredundant = []
	for u,v in combinations( G.nodes(), 2):
	    if adj[u] <= adj[v]:
		#print u, '<', v
		Vredundant.append( u )
		dominator[ u ]=v
	    elif adj[v] <= adj[u]:
		#print v, '<', u
		Vredundant.append( v )
		dominator[ v ]=u
	G.remove_nodes_from(Vredundant)
    return G,dominator
#}
