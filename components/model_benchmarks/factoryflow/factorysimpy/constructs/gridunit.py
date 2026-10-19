from FactorySimPy_constructs.src.factorysimpy.constructs.flowunit import FlowUnit
import simpy, sys, os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import importlib

from factorysimpy.nodes.machine import Machine
from factorysimpy.edges.buffer import Buffer
from factorysimpy.nodes.source import Source
from factorysimpy.nodes.sink import Sink

class GridUnit:
    def __init__(self, id, config, env):
        self.id = id
        self.flowunits = {}
        self.common_source = None
        self.common_sink = None
        self.config = config


        if config.get("common_source"):
            src_cfg = config["common_source"]
            self.common_source = self.create_node(src_cfg["type"], src_cfg["id"], env, **src_cfg.get("params", {}))

        if config.get("common_sink"):
            sink_cfg = config["common_sink"]
            self.common_sink = self.create_node(sink_cfg["type"], sink_cfg["id"], env, **sink_cfg.get("params", {}))


        for fu_entry in config["flowunits"]:
            fu_id = fu_entry["id"]
            fu_config = fu_entry["config"]
            FlowUnit = _load_flowunit_class()
            fu_obj = FlowUnit(fu_id, fu_config)
            self.flowunits[fu_id] = fu_obj

        if self.common_source:
            for fu in self.flowunits.values():
                fu.nodes["source"] = self.common_source
        
        if self.common_sink:
            for fu in self.flowunits.values():
                fu.nodes["sink"] = self.common_sink

        self._connect_shared_nodes()

    def _load_flowunit_class(self):
        module = importlib.import_module('factorysimpy.constructs.flowunit')
        return getattr(module, 'FlowUnit')
    
    def connect_shared_nodes(self):
        for fu_id , fu in self.flowunits.items():
            node_list = [n["id"] for n in fu.config["nodes"]]

            entry_node_id = node_list[0]
            entry_node = fu.nodes[entry_node_id]

            exit_node_id = node_list[-1]
            exit_node = fu.nodes[exit_node_id]

            if self.common_source:
                for edge in fu.edges.values():
                    if isinstance(edge, Buffer) and edge.src_node is None:
                        edge_in = edge
                        break
                edge_in.connect(fu.nodes["source"], entry_node)

            if self.common_sink:
                for edge in fu.edges.values():
                    if isinstance(edge, Buffer) and edge.dst_node is None:
                        edge_out = edge
                        break
                edge_out.connect(exit_node, fu.nodes["sink"])
                            
            