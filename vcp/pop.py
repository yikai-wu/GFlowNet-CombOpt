import os, sys, time
from itertools import combinations
import networkx as nx
import random
from gurobipy import *

def color( G, PRINT=0, TLIM=3600, solution=None): #{
    """ 
    ILP:
    ====
	this is model POP from [1]:

	[1] A.Jabrayilov, P.Mutzel "New Integer Linear Programming Models for the Vertex Coloring Problem"

    CODE VERSION:
    =============
	vcp_pop_10_1.py

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
    #   ___                  
    #  / _ \     _      __ _ 
    # | | | |  _| |_   / _` |
    # | |_| | |_   _| | (_| |
    #  \__\_\   |_|    \__, |
    #                     |_|
    c0 = time.clock()
    Qq = best_clique( G, H, PRINT )
    if PRINT&4:	print 'Time Qq:', (time.clock() - c0)
    c0 = time.clock()

    lb = len(Qq)

    if PRINT&4:	print '(V,E,lb,ub,H) = (%s,%s,%s,%s,%s)' % (len(V),len(E),lb,ub,H)

    if lb == ub: 
	rt   = '%.2F' %(time.clock() - c00)
	return len(V_0), len(E_0), len(V), len(E), len(Qq), H, lb, ub, 'init', rt

    #  __  __           _      _ 
    # |  \/  | ___   __| | ___| |
    # | |\/| |/ _ \ / _` |/ _ \ |
    # | |  | | (_) | (_| |  __/ |
    # |_|  |_|\___/ \__,_|\___|_|
    #                            
    p = Model("coloring")
    p.params.OutputFlag = 1&PRINT
    p.params.SEED       = 1	    # randomized=False
    p.params.THREADS    = 1	    # nur ein thread
    p.params.TimeLimit  = TLIM
    
    #  _   _ 
    # | | | |
    # | |_| |
    # |  _  |
    # |_| |_|
    #        
    H = ub

    # POP variables
    y = { (i,u): p.addVar(vtype=GRB.BINARY) for u in V for i in range(H-1) }
    for u in V: y[ H-1, u ] = p.addVar(0,0)

    #                 _              _      ___  
    #   ___ _ __ ___ | |__   ___  __| |    / _ \ 
    #  / _ \ '_ ` _ \| '_ \ / _ \/ _` |   | | | |
    # |  __/ | | | | | |_) |  __/ (_| |   | |_| |
    #  \___|_| |_| |_|_.__/ \___|\__,_|    \__\_\
    #                                            
    Q = Qq[:-1]
    q = Qq[-1]

    if PRINT&8:	print 'Q:', Q
    if PRINT&8:	print 'q:', q
    for i in range( len(Q)  ): y[ i, Q[i] ] = p.addVar(0,0)
    for i in range(1,len(Q) ): y[ i-1, Q[i] ] = p.addVar(1,1)
    y[ len(Q)-1, q ] = p.addVar(1,1)

    p.update()

    """    _     _           _   _       
      ___ | |__ (_) ___  ___| |_(_)_   __
     / _ \| '_ \| |/ _ \/ __| __| \ \ / /
    | (_) | |_) | |  __/ (__| |_| |\ V / 
     \___/|_.__// |\___|\___|\__|_| \_/  
              |__/                       
    """

    p.setObjective( 
	1 + sum ([ y[i,q] for i in range(H) ])
	, GRB.MINIMIZE
    )

    """        _     _           _   
     ___ _   _| |__ (_) ___  ___| |_ 
    / __| | | | '_ \| |/ _ \/ __| __|
    \__ \ |_| | |_) | |  __/ (__| |_ 
    |___/\__,_|_.__// |\___|\___|\__|
		  |__/               
    """
    # (1)
    for u in V:
	for i in range(H-1): 
	    p.addConstr( y[i,u] - y[i+1,u] >= 0 ) # (1)

    for u in V:
	for i in range(H):    
	    p.addConstr( y[ i, q ] - y[ i, u ] >= 0 ) 

    for u,v in E:
	p.addConstr( y[0,u] + y[0,v] >= 1 ) 
	for i in range(1,H):    
	    p.addConstr( y[i-1,u] - y[i,u] + y[i-1,v] - y[i,v] <= 1 )   # (*3)

    if PRINT&4:	print 'Time write IP:', (time.clock() - c0)
    
    p.optimize()

    rtIP = '%.2F' % p.runtime 
    rt   = '%.2F' %(time.clock() - c00)

    print_solution_to_check_sol_file(dominator,H,y,solution)
    if PRINT&2: print_solution(dominator,H,y,Qq)
    return len(V_0), len(E_0), len(V), len(E), lb, ub, p.objBound, p.objVal, rtIP, rt
#}


def best_clique( G, H, PRINT  ): #{
    t0 = time.clock()
    lenV = len(G.nodes())
    E = { (u,v) for u,v in G.edges() if u != v }
    number_vars = H * lenV
    bestQplusCutQ = 0 
    Q=[]
    max_try=(300*len(E)/lenV) 
    Gcomplement = nx.complement( G )
    for i in range( max_try ):
	random.seed(i)
	_Q=nx.maximal_independent_set( Gcomplement )
	setQ = set(_Q)
	lenQ = len(_Q)
	if lenQ == H:
	    """ lb == ub, so H ist Opt solution """
	    return _Q 
	cutQ = sum([ 1 for u,v in E if len( setQ & {u,v} ) == 1 ]) 
	QplusCutQ = H * lenQ + cutQ
	if bestQplusCutQ < QplusCutQ:
	    bestQplusCutQ = QplusCutQ
	    Q = _Q
	    remaining_number_vars = number_vars - QplusCutQ
	    string='\t|Q|: %s   #Vars: %s/%s     %s sec\r' % ( lenQ, remaining_number_vars, number_vars, time.clock() - t0 )
	    if PRINT&8:
		sys.stderr.write( string );sys.stdout.flush()
	if time.clock() - t0 > 60:
	    break

    if PRINT&4:
	sys.stdout.write( string );sys.stdout.flush()
	print
    return Q
#}



def get_solution(dominator,H,y): #{
    epsilon=0.000000001
    sol={}
    for v in dominator:

	u = dominator[ v ]
	while u != dominator[ u ]:  u = dominator[ u ]

	if abs(y[0,u].x) <= epsilon:
	    sol[ u ] = 0+1
	for i in range(1,H):
	    if abs( y[i,u].x + (1-y[i-1,u].x) ) <= epsilon:
		sol[ u ] = i+1

	sol[ v ] = sol[ u ]

    return sol
#}

def print_solution(dominator,H,y,Q): #{
    """
    This method prints the layout/coloring. It works only if vertices
    are labeled as:

	    V={ '1', '2', ... '|V|' }"
    """
    try:
	sol = get_solution(dominator,H,y)

	bold='\033[1;31m'
	norm='\033[0m'
	for c in sorted(set(sol.values())):
	    print c, '...',
	    for i in range(1,len(sol)+1):
		v = str( i )    # we assume: V={ '1', '2', ... '|V|' }
		if sol[ v ] == c:
		    if v in Q:
			print '%s%s%s' % (bold, v, norm),
		    else: 
			print v,
	    print
	print
    except KeyError:
	print "\nKeyError: print_solution works only if V={ '1', '2', ... '|V|' }"
#}

def print_solution_to_check_sol_file(dominator,H,y,check_sol_file): #{
    if check_sol_file != None:
	sol = get_solution(dominator,H,y)

	os.system( 'echo -n > %s' % check_sol_file )
	for i in range(1,len(sol)+1):
	    v = str( i )    # we assume: V={ '1', '2', ... '|V|' }
	    os.system( 'echo %s >> %s' % ( sol[v], check_sol_file ) )
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
