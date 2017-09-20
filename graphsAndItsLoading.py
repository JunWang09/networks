# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 13:45:32 2017

@author: junwan
"""

import networkx as nx

'''
DiGraph
 - edge
 - node 
'''
G = nx.Graph()
G.add_edge('A','B', weight=6, relation='family')
G.add_edge('B','C', weight=13, relation='friend')
G.add_node('A', role='trader')
G.add_node('B', role='analyst')
G.add_node('C', role='manager')

G.nodes()
G.nodes(data=True)
G.node['A']

G.edges()
G.edges(data=True)
G.edges(data='relation')

G.edge['A']['B']
G.edge['A']['B']['weight']
G.edge['B']['A']['weight']

'''
DiGraph
'''
G=nx.DiGraph()
G.add_edge('A','B', weight=6, relation='family')
G.add_edge('B','C', weight=13, relation='friend')
G.edge['A']['B']['weight']
G.edge['B']['A']['weight']

'''
MultiGraph
'''
G=nx.MultiGraph()
G.add_edge('A','B', weight=6, relation='family')
G.add_edge('A','B', weight=18, relation='friend')
G.add_edge('C','B', weight=13, relation='friend')

G.edge['A']['B']

'''
MultiDiGraph
'''
G=nx.MultiDiGraph()
G.add_edge('A','B', weight=6, relation='family')
G.add_edge('A','B', weight=18, relation='friend')
G.add_edge('C','B', weight=13, relation='friend')
G.edge['A']['B'][0]

'''
bipartite
 - for example: Fans vs basketball teams
'''
from networkx.algorithms import bipartite
B=nx.Graph()
B.add_nodes_from(['A','B','C','D','E'], bipartite=0)
B.add_nodes_from([1,2,3,4], bipartite=1)
B.add_edges_from([('A',1),('B',1),('C',1),('C',3),('D',2),('E',3),('E',4)])

bipartite.is_bipartite(B)
X =set([1,2,3,4])
bipartite.is_bipartite_node_set(B,X)
X =set([1,2,3,4,'A'])
bipartite.is_bipartite_node_set(B,X)
bipartite.sets(B)

B.add_edge('A','B')
B.remove_edge('A', 'B')


B=nx.Graph()
B.add_edges_from([('A',1),('B',1),('C',1),('D',1),('H',1),('B',2),('C',2),('D',2),('E',2),('G',2),('E',3),('F',3),('H',3),('J',3),('E',4),('I',4),('J',4)])
X = set(['A','B','C','D','E','F','G','H','I','J'])
P = bipartite.projected_graph(B,X)

X = set([1,2,3,4])
P = bipartite.projected_graph(B,X)
P = bipartite.weighted_projected_graph(B,X)
P.edges(data=True)


'''
Loading Graphs in NetworkX
'''
import networkx as nx
import numpy as np
import pandas as pd
%matplotlib inline
import os
os.chdir('E:/')
# Instantiate the graph
G1 = nx.Graph()
# add node/edge pairs
G1.add_edges_from([(0, 1),
                   (0, 2),
                   (0, 3),
                   (0, 5),
                   (1, 3),
                   (1, 6),
                   (3, 4),
                   (4, 5),
                   (4, 7),
                   (5, 8),
                   (8, 9)])

# draw the network G1
nx.draw_networkx(G1)

'''
Adjacency List

G_adjlist.txt is the adjaceny list representation of G1.
It can be read as follows:
0 1 2 3 5  →  node 0 is adjacent to nodes 1, 2, 3, 5
1 3 6  →  node 1 is (also) adjacent to nodes 3, 6
2  →  node 2 is (also) adjacent to no new nodes
3 4  →  node 3 is (also) adjacent to node 4
and so on. 
Note that adjacencies are only accounted for once (e.g. node 2 is adjacent to node 0, but node 0 
is not listed in node 2's row, because that edge has already been accounted for in node 0's row).

G_adjlist.txt
0 1 2 3 5
1 3 6
2
3 4
4 5 7
5 8
6
7
8 9
9
'''
G2 = nx.read_adjlist('G_adjlist.txt', nodetype=int)
G2.edges()

'''
Adjacency Matrix

The elements in an adjacency matrix indicate whether pairs of vertices are adjacent or not in the graph. 
Each node has a corresponding row and column. For example, row 0, column 1 corresponds to the edge 
between node 0 and node 1.
Reading across row 0, there is a '1' in columns 1, 2, 3, and 5, which indicates that node 0 is adjacent 
to nodes 1, 2, 3, and 5
'''
G_mat = np.array([[0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
                  [1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
                  [1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
    
G3 = nx.Graph(G_mat)
G3.edges()    
    

'''
Edgelist

The edge list format represents edge pairings in the first two columns. Additional edge attributes 
can be added in subsequent columns. Looking at G_edgelist.txt this is the same as the original graph G1, 
but now each edge has a weight.
For example, from the first row, we can see the edge between nodes 0 and 1, has a weight of 4.

G_edgelist.txt
0 1 4
0 2 3
0 3 2
0 5 6
1 3 2
1 6 5
3 4 3
4 5 1
4 7 2
5 8 6
8 9 1
'''
G4 = nx.read_edgelist('G_edgelist.txt', data=[('Weight', int)])
G4.edges(data=True)

'''
Pandas DataFrame

Graphs can also be created from pandas dataframes if they are in edge list format.
'''
G_df = pd.read_csv('G_edgelist.txt', delim_whitespace=True, header=None, names=['n1', 'n2', 'weight'])

G5 = nx.from_pandas_dataframe(G_df, 'n1', 'n2', edge_attr='weight')
G5.edges(data=True)

'''
Chess Example

Now let's load in a more complex graph and perform some basic analysis on it.
We will be looking at chess_graph.txt, which is a directed graph of chess games in edge list format.

chess_graph.txt
1 2 0	885635999.999997
1 3 0	885635999.999997
1 4 0	885635999.999997
1 5 1	885635999.999997
1 6 0	885635999.999997
7 8 0	885635999.999997
7 9 1	885635999.999997
7 10 1	885635999.999997

Each node is a chess player, and each edge represents a game. The first column with an outgoing edge corresponds to the white player, the second column with an incoming edge corresponds to the black player.
The third column, the weight of the edge, corresponds to the outcome of the game. A weight of 1 indicates white won, a 0 indicates a draw, and a -1 indicates black won.
The fourth column corresponds to approximate timestamps of when the game was played.
We can read in the chess graph using read_edgelist, and tell it to create the graph using a nx.MultiDiGraph.

'''

chess = nx.read_edgelist('chess_graph.txt', data=[('outcome', int), ('timestamp', float)], create_using=nx.MultiDiGraph())
chess.is_directed(), chess.is_multigraph()
chess.edges(data=True)

'''
Looking at the degree of each node, we can see how many games each person played. A dictionary is returned where each key is the player, and each value is the number of games played.
'''
games_played = chess.degree()
games_played

max_value = max(games_played.values())
max_key, = [i for i in games_played.keys() if games_played[i] == max_value]

print('player {}\n{} games'.format(max_key, max_value))
df = pd.DataFrame(chess.edges(data=True), columns=['white', 'black', 'outcome'])
'''
   white	black	outcome
0	  1	    2	{'outcome': 0, 'timestamp': 885635999.999997}
1	  1	    3	{'outcome': 0, 'timestamp': 885635999.999997}
'''

df['outcome'] = df['outcome'].map(lambda x: x['outcome'])

won_as_white = df[df['outcome']==1].groupby('white').sum()
won_as_black = -df[df['outcome']==-1].groupby('black').sum()
win_count = won_as_white.add(won_as_black, fill_value=0)
win_count.head()

win_count.nlargest(5, 'outcome')

