import simpy, sys, os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import importlib

from factorysimpy.nodes.machine import Machine
from factorysimpy.edges.buffer import Buffer
from factorysimpy.nodes.source import Source
from factorysimpy.nodes.sink import Sink

class FlowUnit:
    def __init__(self, id, config):
        self.id = id
        self.nodes = {}
        self.edges = {}

        for n in config["nodes"]:
            self.nodes[n["id"]] = self.create_node(n["type"], n["id"], **n.get("params", {}))

        for e in config["edges"]:
            self.edges[e["id"]] = self.create_edge(e["type"], e["id"], **e.get("params", {}))

        for e in config["edges"]:
            src_node_id = e["params"]["src_node"]
            dest_node_id = e["params"]["dest_node"]
            self.connect_nodes(src_node_id, dest_node_id, e["id"])

        
    def create_node(self,node_type, node_id, **params):
        module = importlib.import_module(f'factorysimpy.nodes.{node_type.lower()}')
        cls = getattr(module, node_type)
        print(type(params))
        return cls(env,node_id, **params)
    
    def create_edge(self, edge_type, edge_id, **params):
        module = importlib.import_module(f'factorysimpy.edges.{edge_type.lower()}')
        cls = getattr(module, edge_type)

        # Convert node names to actual node objects
        src_node_name = params.pop("src_node")
        dest_node_name = params.pop("dest_node")

        src_node = self.nodes[src_node_name]
        dest_node = self.nodes[dest_node_name]

        print(f"Creating edge {edge_id}: {edge_type} from {src_node_name} → {dest_node_name}")

        # Pass src, dest INTO constructor
        return cls(env, edge_id)

    
    def connect_nodes(self, src_node_id, dest_node_id, edge_id):
        edge = self.edges[edge_id]
        src_node = self.nodes[src_node_id]
        dest_node = self.nodes[dest_node_id]
        edge.connect(src_node, dest_node)


    def delete_node(self, node_id):
        if node_id not in self.nodes:
            raise ValueError(f"Node with id {node_id} does not exist.")
        if node_id in self.nodes:
            in_edges = self.nodes[node_id].in_edges
            out_edges = self.nodes[node_id].out_edges
            del self.nodes[node_id]
            for edge in in_edges + out_edges:
                if edge.id in self.edges:
                    del self.edges[edge.id]
    
    def delete_edge(self, edge_id):
        if edge_id not in self.edges:
            raise ValueError(f"Edge with id {edge_id} does not exist.")
        if edge_id in self.edges:
            del self.edges[edge_id]

    def update_node(self, node_id, **params):
        if node_id not in self.nodes:
            raise ValueError(f"Node with id {node_id} does not exist.")
        
        node = self.nodes[node_id]

        for key, value in params.items():
            if hasattr(node, key):
                setattr(node, key, value)
            else:
                raise ValueError(f"Node does not have attribute '{key}'.")
            
    def update_edge(self, edge_id, **params):
        if edge_id not in self.edges:
            raise ValueError(f"Edge with id {edge_id} does not exist.")
        
        edge = self.edges[edge_id]

        for key, value in params.items():
            if hasattr(edge, key):
                setattr(edge, key, value)
            else:
                raise ValueError(f"Edge does not have attribute '{key}'.")

    def __repr__(self):
        return f"FlowUnit(id={self.id})"
    

if __name__ == "__main__":
    env = simpy.Environment()   
    config = {
        "nodes": [
            {"id": "source", "type": "Source", "params": { "inter_arrival_time": 1.5}},
            {"id": "machine", "type": "Machine", "params": {"processing_delay": 2.0, "blocking": True,"work_capacity":1}},
            {"id": "sink", "type": "Sink", "params": {}},
        ],
        "edges": [
            {"id": "buffer1", "type": "Buffer", "params": {"src_node": "source", "dest_node": "machine"}},
            {"id": "buffer2", "type": "Buffer", "params": {"src_node": "machine", "dest_node": "sink"}}
        ], 

    }

    flow_unit = FlowUnit("FU1", config)

    for node_id, node in flow_unit.nodes.items():
        print(f"Created node: {node.__class__.__name__} with ID: {node_id}")
        for attr, value in node.__dict__.items():
            print(f"  {attr}: {value}")
    
    for edge_id, edge in flow_unit.edges.items():
        print(f"Created edge: {edge.__class__.__name__} with ID: {edge_id}")
        for attr, value in edge.__dict__.items():
            print(f"  {attr}: {value}")