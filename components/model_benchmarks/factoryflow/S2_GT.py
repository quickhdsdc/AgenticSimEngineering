"""System has two machines M1and M2 with delay 2. 
 Source (ID= SRC) with inter arrival time of 1.2. 
 SRC is connected to M1 and M1 to M2 and M2 to Sink via Buffers, 
B_src_1, B_1_2, B_2_3. All buffers have a capacity of 2."""


import simpy
#import factorysimpy
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','src')))

from factorysimpy.nodes.node import Node
from factorysimpy.nodes.machine import Machine
from factorysimpy.edges.buffer import Buffer
from factorysimpy.nodes.source import Source
from factorysimpy.nodes.sink import Sink


                    


class LinearModel(Node):
    def __init__(self, env, id, n=5):
        super().__init__(env, id)
        self.SRC = Source(env, id="SRC")
        self.add_child_node(self.SRC)
        self.M = [Machine(env, id=f"M{i+1}") for i in range(n)]
               
        self.add_child_node(self.M)
        self.SINK = Sink(env, id="sink")
        self.add_child_node(self.SINK)

        
        # edges
        self.B0=Buffer(env, id=f"B_src_1")
        self.Bsink=Buffer(env, id=f"B_{n}_sink")
        self.edgs = [Buffer(env, id=f"B_{i+1}_{i+2}") for i in range(n-1)]
        self.BUFFER=[self.B0]+self.edgs+[self.Bsink]
        self.add_child_edge(self.BUFFER)
        # adding nodes and edges to parent node

        self.BUFFER[0].connect(self.SRC, self.M[0])
        for i in range(n-1):
            self.BUFFER[i+1].connect(self.M[i],self.M[i+1])
        self.BUFFER[-1].connect(self.M[-1], self.SINK)

        
        

env= simpy.Environment()
TOP = LinearModel(env,"TOP",n=2) 

TOP.SRC.inter_arrival_time = 1.2
for i in range(len(TOP.M)):
    TOP.M[i].processing_delay = 2

for i in range(len(TOP.BUFFER)):
    TOP.BUFFER[i].capacity = 2



TOP.fill_hierarchical_id()
from factorysimpy.utils import utils



fig = utils.draw_blockdiagram(TOP)
path = fig.render(filename="S2_GT", directory=".", format="png", cleanup=True)

TOP.validate()
TOP.show_hierarchy()
TOP.run_simulation(25)

        









