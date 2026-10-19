"""System has a single machine(ID=MACHINE) with delay 2.  
Source (ID= SRC) with inter arrival time of 1.2. 
SRC is connected to Machine via a Buffer(ID=BUFFER1) and MACHINE to
 SINK via a Buffer(ID=BUFFER2).
Buffer has capacity of 3. MACHINE has a work_capacity is 2"""


import simpy
#import factorysimpy
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','src')))
from factorysimpy.utils.utils import draw_blockdiagram
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
        self.MACHINE = Machine(env, id=f"MACHINE") 
               
        self.add_child_node(self.MACHINE)
        self.SINK = Sink(env, id="SINK")
        self.add_child_node(self.SINK)

        
        # edges
        self.BUFFER = [Buffer(env, id=f"BUFFER{i+1}") for i in range(n+1)]
        self.add_child_edge(self.BUFFER)
        # adding nodes and edges to parent node

        self.BUFFER[0].connect(self.SRC, self.MACHINE)
        for i in range(n-1):
            self.BUFFER[i+1].connect(self.MACHINE,self.MACHINE)
        self.BUFFER[n].connect(self.MACHINE, self.SINK)

        
        

env= simpy.Environment()
TOP = LinearModel(env,"TOP",n=1) 

TOP.fill_hierarchical_id() 
TOP.fill_hierarchical_id()  
TOP.MACHINE.processing_delay = 2
TOP.MACHINE.work_capacity = 2
TOP.SRC.inter_arrival_time = 1.2
TOP.BUFFER[0].capacity = 3
TOP.BUFFER[1].capacity = 3
fig = draw_blockdiagram(TOP)
fig.render(filename="S1_GT", directory=".", format="png", cleanup=True)
TOP.validate()
TOP.show_hierarchy()
TOP.run_simulation(25)

        









