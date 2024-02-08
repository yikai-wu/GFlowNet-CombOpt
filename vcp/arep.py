import os, sys, time
from itertools import combinations
import networkx as nx
import random
from gurobipy import *

def color( G, PRINT=0, TLIM=3600, solution=None): #{
    """ 
    ALGORITHM:
    ==========
        [1] Campelo at all: "On the asymmetric representatives formulation for the VCP", 2007

    CODE VERSION
    ==============
	- this is vcp_rep_asym.py

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
    Gcomplement=nx.complement( G )

    if PRINT&4: print 'Reduction (V,E): (%s,%s) --> (%s,%s)    %0.2F sec  ' % (
		len(V_0), len(E_0), len(V), len(E), (time.clock() - c00)); 


    ub = len(set(nx.greedy_color(G).values()))
    #  _   _ 
    # | | | |
    # | |_| |
    # |  _  |
    # |_| |_|   Achtung: dieser IP braucht kein H, aber wegen fairplay test: lb == ub 
    #        
    H = ub
    #  ____   
    # / ___|  
    # \___ \  
    #  ___) | 
    # |____/  
    #         
    #  __   _     ____    __ 
    # | _| / |   / ___|  |_ |
    # | |  | |   \___ \   | |
    # | |  | |_   ___) |  | |
    # | |  |_( ) |____/   | |
    # |__|   |/          |__|
    c0 = time.clock()
    S = best_clique( G, PRINT )
    if PRINT&8:	print 'S:', S
    if PRINT&4:	print 'Time S:', (time.clock() - c0)
    c0 = time.clock()

    lb = len(S)

    if PRINT&4:	print '(V,E,lb,ub,H) = (%s,%s,%s,%s,%s)' % (len(V),len(E),lb,ub,H)

    if lb == ub: 
	rt   = '%.2F' %(time.clock() - c00)
	return len(V_0), len(E_0), len(V), len(E), len(S), H, lb, ub, 'init', rt

    # orderin for vertices for asymmetric formulation
    Vordered = S + list( set(V) - set(S) )
    pi={Vordered[i]:i for i in range(len(V))}

    #  __   _     _____   __ 
    # | _| / |   |_   _| |_ |
    # | |  | |     | |    | |
    # | |  | |_    | |    | |
    # | |  |_( )   |_|    | |
    # |__|   |/          |__|
    T=set()
    for t in V: 
        T.add( t )
        for w in Gcomplement.neighbors(t):
            if pi[t] < pi[w]:
                T.remove( t )
                break

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
    

    # REP variables
    y = { (u,u): p.addVar(vtype=GRB.BINARY) for u in V }
    for u,v in Gcomplement.edges():
        if pi[u] < pi[v]:
            y[ u,v ] = p.addVar(vtype=GRB.BINARY) 
            y[ v,u ] = p.addVar(0,0)
        else:
            y[ v,u ] = p.addVar(vtype=GRB.BINARY) 
            y[ u,v ] = p.addVar(0,0)
    #                 _              _   ____  
    #   ___ _ __ ___ | |__   ___  __| | / ___| 
    #  / _ \ '_ ` _ \| '_ \ / _ \/ _` | \___ \ 
    # |  __/ | | | | | |_) |  __/ (_| |  ___) |
    #  \___|_| |_| |_|_.__/ \___|\__,_| |____/ 
    #                                          
    for s in S: y[ s, s ] = p.addVar(1,1)
    p.update()

    """    _     _           _   _       
      ___ | |__ (_) ___  ___| |_(_)_   __
     / _ \| '_ \| |/ _ \/ __| __| \ \ / /
    | (_) | |_) | |  __/ (__| |_| |\ V / 
     \___/|_.__// |\___|\___|\__|_| \_/  
              |__/                       
    """

    p.setObjective( 
	len(S) + sum ([ y[v,v] for v in set(V)-set(S) ])
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
	p.addConstr( y[u,u] + sum([ y[v,u] for v in Gcomplement.neighbors(u) ]) >= 1 )

    # [1, Clicue K with size 1 and 2]
    for u in set(V)-T:
        # |K|=1:
        for w in Gcomplement.neighbors(u):
            p.addConstr( y[u,w] <= y[u,u] )
        # |K|=2:
	for v,w in E:
	    if len( {v,w} & set(Gcomplement.neighbors(u)) ) > 1 :
		p.addConstr( y[u,v] + y[u,w] <= y[u,u] )

    if PRINT&4:	print 'Time write IP:', (time.clock() - c0)
    
    p.optimize()

    rtIP = '%.2F' % p.runtime 
    rt   = '%.2F' %(time.clock() - c00)

    if PRINT&2:	print_solution(dominator,y,S)
    print_solution_to_check_sol_file(dominator,y,solution)

    return len(V_0), len(E_0), len(V), len(E), lb, ub, p.objBound, p.objVal, rtIP, rt
#}


def best_clique( G, PRINT  ): #{
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
	    if PRINT&8:
		sys.stderr.write( string );sys.stdout.flush()
	if time.clock() - t0 > 60:
	    break

    if PRINT&4:
	sys.stdout.write( string );sys.stdout.flush()
	print
    return Q
#}

def get_solution(dominator,y): #{
    epsilon=0.000000001
    sol={}
    color=0
    for v in dominator:
	u = dominator[ v ]
	while u != dominator[ u ]:  
	    u = dominator[ u ]
	if u not in sol:	# consider each domiantor only 1 time
	    if abs(1 - y[u,u].x) <= epsilon:
		color += 1;
		sol[ u ] = color

    for (u,v) in y:
	if u not in sol:	# not owerride sol[u] corresponding to y[u,u]==1
	    if abs(1 - y[v,u].x) <= epsilon:
		sol[ u ] = sol[ v ]
	if v not in sol:	# not owerride sol[w] corresponding to y[w,w]==1
	    if abs(1 - y[u,v].x) <= epsilon:
		sol[ v ] = sol[ u ]
		#print '----', w, sol[w]

    for v in dominator:
	u = dominator[ v ]
	while u != dominator[ u ]:  u = dominator[ u ]
	
	if v != u:
	    sol[ v ] = sol[ u ]
    
#    for c in range(1,len(sol)+1):
#	print c, sol[ str(c) ]
    return sol 
#}

def print_solution(dominator,y,Q): #{
    """
    This method prints the layout/coloring. It works only if vertices
    are labeled as:

	    V={ '1', '2', ... '|V|' }"
    """
    try:
	sol = get_solution(dominator,y)

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

def print_solution_to_check_sol_file(dominator,y,check_sol_file): #{
    if check_sol_file != None:
	sol = get_solution(dominator,y)

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