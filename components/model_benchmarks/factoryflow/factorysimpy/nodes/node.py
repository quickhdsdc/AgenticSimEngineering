import simpy
from collections import OrderedDict
from graphviz import Digraph



class Node:
    """     
    Base class to represent an active entity in a manufacturing system,
    such as machines, splits, or joints.

    Parameters:
        id (str): Identifier for the node.
        node_setup_time ( int, or float, optional): Initial setup time for the node. Can be:
        
            - int or float: Used as a constant delay. 
        in_edges (list, optional): List of input edges connected to the node. 
        out_edges (list, optional): List of output edges connected to the node. 

    Raises:
        TypeError: If the type of `env` or `id` is incorrect.
        ValueError: If `node_setup_time` input is invalid.
    """
    
    def __init__(self,env,id, in_edges = None, out_edges = None, node_setup_time= 0):
   
        # Type checks
        if not isinstance(env, simpy.Environment):
            raise TypeError("env must be a simpy.Environment instance")
        if not isinstance(id, str):
            raise TypeError("name must be a string")
        self.env = env
        self.id = id # Identifier for the node.
        self.parent = None # Identifier for the parent. Set as None for Top-level nodes.
        self.hierarchical_id = ""
        self.child_nodes ={}
        self.child_edges ={}
        self.node_setup_time = node_setup_time # Time taken to set up the node.
        self.in_edges = in_edges # List of input edges connected to the node.
        self.out_edges = out_edges #List of output edges connected to the node.

       
        if isinstance(node_setup_time, (int, float)):
            self.node_setup_time = node_setup_time
        else:
            raise ValueError(
                "Invalid node_setup_time value. Provide constant ( int or float)."
            )
        
       
    def get_delay(self,delay):
        """
        Returns value based on the type of parameter `delay` provided.

        Args:
             delay (int, float, generator, or callable): The delay time, which can be:
             
                - int or float: Used as a constant delay.
                - generator: A generator instance yielding delay values.
                - callable: A function that returns a delay values.

        Returns:
               Returns a constant delay if `delay` is an int or float, a value yielded  if `delay` is a generator, or the value returned from a Callable function if `delay` is callable.
        """
        if hasattr(delay, '__next__'):
            # Generator instance
            val = next(delay)
        elif callable(delay):
            # Function
            val = delay()
        else:
            # int or float
            val = delay
        #print(self.id)
        assert val >= 0, f"{self.id}- Delay must be non-negative"
        return val
    
    def update_state(self, new_state: str, current_time: float):
        """
        Update node state and track the time spent in the previous state.
        
        Args:
            new_state (str): The new state to transition to. Must be one of "SETUP_STATE", "GENERATING_STATE", "BLOCKED_STATE".
            current_time (float): The current simulation time.

        """
        
        if self.state is not None and self.stats["last_state_change_time"] is not None:
            elapsed = current_time - self.stats["last_state_change_time"]

            self.stats["total_time_spent_in_states"][self.state] = (
                self.stats["total_time_spent_in_states"].get(self.state, 0.0) + elapsed
            )
        self.state = new_state
        self.stats["last_state_change_time"] = current_time

    def add_in_edges(self, edge):
        #Override this method in subclasses.
        raise NotImplementedError("add_in_edges must be implemented in a subclass.")

    def add_out_edges(self):
        
        #Override this method in subclasses
        
        raise NotImplementedError("add_out_edges must be implemented in a subclass.")

    def behaviour(self):
        #Override this method in subclasses
        
        raise NotImplementedError("behaviour must be implemented in a subclass.")

    
    def add_child_node(self, child_node):
        """
        Add Node / Machine / Source / Sink (subclasses of Node) as children.
        Uses class name string matching instead of isinstance()
        to avoid cross-file dependency problems.
        """

        # Accepted class names for nodes
        valid_node_classes = {
            "Node", "Machine", "Source", "Sink", "Processor", "Splitter", "Combiner"
        }

        def is_node_like(obj):
            """Check Node-ness using class name string."""
            return obj.__class__.__bases__[0].__name__ in valid_node_classes
            #return obj.__class__.__name__ in valid_node_classes

        def add_single_node(n, is_from_list=False):
            if not is_node_like(n):
                raise TypeError(
                    f"Child node must be Node-like (class name in {valid_node_classes}), "
                    f"but got {n.__class__.__name__}"
                )

            # Prevent duplicates
            if n.id in self.child_nodes:
                raise ValueError(f"Node '{n.id}' already added as child of '{self.id}'")

            # Assign parent
            n.parent = self

            # Allow attribute access only for single nodes, not for list items
            # This prevents setattr() from overwriting list variables
            if not is_from_list:
                setattr(self, n.id, n)

            # Store
            self.child_nodes[n.id] = n

        # Case 1: single node-like object
        if not isinstance(child_node, list):
            add_single_node(child_node, is_from_list=False)
            return

        # Case 2: list / nested lists
        for item in child_node:
            if not isinstance(item, list):
                add_single_node(item, is_from_list=True)
            else:
                for sub in item:
                    add_single_node(sub, is_from_list=True)




    def add_child_edge(self, child_edge):
        """
        Add Edge / Buffer / Conveyor / Transporter objects as child edges.
        Uses class name string matching to avoid cross-file circular dependencies.
        """

        valid_edge_classes = {
            "Edge", "Buffer", "Conveyor", "Transporter", "Belt", "Queue"
        }

        def is_edge_like(obj):
            return obj.__class__.__name__ in valid_edge_classes

        def add_single_edge(e):
            if not is_edge_like(e):
                raise TypeError(
                    f"Child edge must be Edge-like (class name in {valid_edge_classes}), "
                    f"but got {e.__class__.__name__}"
                )

            # avoid duplicates
            if e.id in self.child_edges:
                raise ValueError(f"Edge '{e.id}' already added as child of '{self.id}'")

            # assign parent
            e.parent = self

            # dynamic attribute access
            setattr(self, e.id, e)

            # store edge
            self.child_edges[e.id] = e

        # Case 1: single edge object
        if not isinstance(child_edge, list):
            add_single_edge(child_edge)
            return

        # Case 2: list or nested list
        for item in child_edge:
            if not isinstance(item, list):
                add_single_edge(item)
            else:
                for sub in item:
                    add_single_edge(sub)




    # -------------------------
    # Hierarchy / flattening
    # -------------------------
    def show_hierarchy(self, level: int = 0):
        """Pretty-print the tree of nodes and immediate edges for quick debugging."""
        indent = "  " * level
        print(f"{indent}Node ID: {self.id}")
        for child_node in self.child_nodes.values():
            child_node.show_hierarchy(level + 1)
        for child_edge in self.child_edges.values():
            # safe attribute access (edge may be duck-typed)
            fn = getattr(child_edge, "src_node", None)
            tn = getattr(child_edge, "dest_node", None)
            fn_id = fn.id if fn is not None else "?"
            tn_id = tn.id if tn is not None else "?"
            print(f"{indent}  Edge ID: {child_edge.id} (from {fn_id} to {tn_id})")

    def get_all_nodes_edges(self):
        """
        Return flattened (deterministic) dictionaries of all nodes and edges under this node.
        Returns:
            nodes: OrderedDict[node_id -> Node]
            edges: OrderedDict[edge_id -> Edge]
        """
        nodes = OrderedDict()
        edges = OrderedDict()

        # include self first
        nodes[self.hierarchical_id] = self

        # collect recursively from child nodes
        for child in self.child_nodes.values():
            cnodes, cedges = child.get_all_nodes_edges()
            # update in deterministic order
            for nid, nobj in cnodes.items():
                if nid not in nodes:
                    nodes[nid] = nobj
            for eid, eobj in cedges.items():
                if eid not in edges:
                    edges[eid] = eobj

        # collect direct child edges (avoid duplicates)
        for edge in self.child_edges.values():
            if edge.id not in edges:
                edges[edge.hierarchical_id] = edge

        return nodes, edges
    
    def extract_model_data(self):
        """
        Build a dictionary representation of the full model:
        
            {
            "nodes": { node_id: {"type": ClassName, "params": {...}}, ... },
            "edges": { edge_id: {"type": ClassName, "params": {...}}, ... }
            }

        Parameters:
            nodes_flat: dict[node_id -> NodeObject]
            edges_flat: dict[edge_id -> EdgeObject]

        Returns:
            dict
        """

        def extract_params(obj):
            """
            Extract all non-private, non-callable, non-structural attributes
            from any node/edge object.
            """
            params = {}

            skip_keys = {
                "parent", "child_nodes", "child_edges",
                "in_edges", "out_edges", "env",
                "from_node", "to_node", "state", "stats", "inbuiltstore"
            }

            for key, val in obj.__dict__.items():
                if key.startswith("_"): 
                    continue
                if key in skip_keys:
                    continue
                if callable(val):
                    continue
                params[key] = val

            return params

        # MAIN OUTPUT
        output = {
            "nodes": {},
            "edges": {}
        }

        # ------------------------------------
        # PROCESS NODES
        # ------------------------------------
        nodes_flat, edges_flat = self.get_all_nodes_edges()
        for node_id, node_obj in nodes_flat.items():
            output["nodes"][node_id] = {
                "type": node_obj.__class__.__name__,
                "params": extract_params(node_obj)
            }

        # ------------------------------------
        # PROCESS EDGES
        # ------------------------------------
        for edge_id, edge_obj in edges_flat.items():
            output["edges"][edge_id] = {
                "type": edge_obj.__class__.__name__,
                "params": extract_params(edge_obj)
            }

        return output

    
    def update_params(self,obj, params):

        for param_key, param_value in params.items():
            if hasattr(obj, param_key):
                setattr(obj, param_key, param_value)
            else:
                raise AttributeError(f"⚠️ Warning: {obj.id} has no attribute '{param_key}', skipping update.")

    def __str__(self):
        s=""
        #s+=str(self.id)
        s+=str(self.hierarchical_id)
        return s

    def fill_hierarchical_id(self, parent_hierarchical_id=""):
        self.hierarchical_id = f"{parent_hierarchical_id}.{self.id}" if parent_hierarchical_id else str(self.id)
        for child in self.child_nodes.values():
            #print(child.id)
            child.fill_hierarchical_id(self.hierarchical_id)
        for edge in self.child_edges.values():
            #print(edge.id)
            edge.hierarchical_id = f"{self.hierarchical_id}.{edge.id}"
            #edge.fill_hierarchical_id(self.hierarchical_id)

    def get_hierarchical_info(self):
        s= self.__str__()
        for n in self.child_nodes:
            #print(self, s)
            s+="-"
            s+=self.child_nodes[n].get_hierarchical_info()
        return s
    
    
    def stop_behaviour(self):
        # Logic to stop the simulation
        if hasattr(self, "behave"):
            if self.behave.is_alive:
                try:
                    self.behave.interrupt()
                    for child in self.child_nodes.values():
                        child.stop_behaviour()
                except Exception as e:
                    print(f"{self.env.now}: Error stopping machine behaviour: {e}")
        

    def draw_blockdiagram(self, output_file="factory_blockdiagram"):
        """
        Build a Graphviz block diagram from the dictionary produced by extract_model_data().
        
        model_dict format:
        {
            "nodes": {
                node_id: {"type": "<ClassName>", "params": {...}},
                ...
            },
            "edges": {
                edge_id: {"type": "<ClassName>", "params": {...}},
                ...
            }
        }

        Notes:
        - Nodes and Edges are drawn as rectangles.
        - Connections are inferred from:
            edge.src_node.id  → edge.id → edge.dest_node.id
        """
        model_dict = self.extract_model_data()
        #print("nodes are", model_dict["nodes"])
        #print("edges are", model_dict["edges"])
        dot = Digraph(comment="Factory Model Block Diagram", format="png")
        #dot.graph_attr["dpi"] = "100"
        dot.graph_attr["size"] = "16,14"
        dot.attr(rankdir="LR", fontsize="12", nodesep="0.5", ranksep="0.75")

        created_nodes = set()

        # -------------------------
        # Helper: Add a rectangle node
        # -------------------------
        def add_node(node_id, params, shape="box", color="white"):
            hierarchical_id= params.get("hierarchical_id", None)
            if hierarchical_id is None:
                raise ValueError(f"Node '{node_id}' missing 'hierarchical_id' in params.")
            print(info["params"])
            if params["hierarchical_id"] in created_nodes:
                #print(f"Node {node_id} already created, skipping.")
                return
            tooltip = "\n".join([f"{k}: {v}" for k, v in params.items()]) if params else "No params"
            dot.node(hierarchical_id, params["id"], shape=shape, style="filled", fillcolor=color, tooltip=tooltip)
            created_nodes.add(hierarchical_id)
    
          

        # -------------------------
        # Add all model nodes
        # -------------------------
        for node_id, info in model_dict["nodes"].items():
            #print(node_id, info)

            if info["type"] not in ["Machine", "Source", "Sink", "Splitter", "Combiner"]:

                continue
            print("adding_node", info["params"]["hierarchical_id"])
            add_node(node_id, info["params"], color="palegreen")
            #add_node(node_id, info["params"] )

        # -------------------------
        # Add all edges as rectangles
        # -------------------------
        for edge_id, info in model_dict["edges"].items():
            print("adding_edge", info["params"]["hierarchical_id"])
            add_node(edge_id, info["params"], color="lightyellow")
            #add_node(edge_id, info["params"])

        # -------------------------
        # Add connections (NODE → EDGE → NODE)
        # Using runtime objects from model (we need to map them)
        # -------------------------
        # We need access to actual node/edge objects to read src_node/dest_node
        # So model_dict must also include "obj_ref" inside params OR
        # we directly use the original edges_flat (best approach)

        if "edges" in model_dict:
            # BEST: User passed edges_flat for connection reconstruction
            edges_flat = model_dict["edges"]

            for edge_id, edge_obj in edges_flat.items():
                #print(edge_obj)
                if 'src_node' not in edge_obj["params"] or 'dest_node' not in edge_obj["params"]:
                    print(f"⚠️ Warning: Edge '{edge_id}' missing src_node or dest_node attributes, skipping connection.")
                    raise ValueError(f"Incomplete edge: {edge_id}")
                

                src_id = edge_obj['params']['src_node'].hierarchical_id
                dst_id = edge_obj['params']['dest_node'].hierarchical_id

                # Ensure rectangles exist
                #add_node(src_id, model_dict["nodes"][src_id]["params"], "lightgreen")
                #add_node(edge_id, model_dict["edges"][edge_id]["params"], "lightyellow")
                #add_node(dst_id, model_dict["nodes"][dst_id]["params"], "lightgreen")

                # Draw arrows
                dot.edge(src_id, edge_id, color="gray40")
                dot.edge(edge_id, dst_id, color="gray40")

        else:
            print("⚠️ Warning: model_dict must contain 'edges_flat' to reconstruct connections.")

        return dot






    # -------------------------
    # Validation (duplicates, basic connectivity)
    # -------------------------
    def validate(self, verbose: bool = True):
        """
        Validate this node subtree.

        Checks:
          - duplicate node IDs across the subtree (error)
          - duplicate edge IDs across the subtree (error)
        Cycles are allowed (no cycle detection).

        Returns:
            dict with keys: ok (bool), node_duplicates (list), edge_duplicates (list)
        """
        nodes, edges = self.get_all_nodes_edges() # returns a dict of node and edge objects with hierarchical IDs as keys
        node_ids = list(nodes.keys())
        edge_ids = list(edges.keys())

        # detect duplicates by comparing lengths (but get_all_nodes_edges already deduped by id)
        # to be robust, check if any id appears multiple times when traversing original structure:
        seen_nodes = {}
        node_dups = []
        def _walk_for_node_dups(node):
            if node.id in seen_nodes:
                node_dups.append(node.id)
            else:
                seen_nodes[node.id] = node
            for cn in node.child_nodes.values():
                _walk_for_node_dups(cn)
        _walk_for_node_dups(self)

        seen_edges = {}
        edge_dups = []
        def _walk_for_edge_dups(node):
            for e in node.child_edges.values():
                if e.id in seen_edges:
                    edge_dups.append(e.id)
                else:
                    seen_edges[e.id] = e
            for cn in node.child_nodes.values():
                _walk_for_edge_dups(cn)
        _walk_for_edge_dups(self)

        ok = (len(node_dups) == 0) and (len(edge_dups) == 0)
        if verbose:
            if ok:
                print(f"Validation OK: No duplicate node/edge IDs under '{self.id}'.")
            else:
                if node_dups:
                    print(f"Duplicate node IDs found under '{self.id}': {sorted(set(node_dups))}")
                if edge_dups:
                    print(f"Duplicate edge IDs found under '{self.id}': {sorted(set(edge_dups))}")

        return {"ok": ok, "node_duplicates": sorted(set(node_dups)), "edge_duplicates": sorted(set(edge_dups))}

    
    @property
    def top(self):
        """Return the top-level parent node in the hierarchy."""
        current = self
        while current.parent is not None and current.parent is not current: # to prevent cycles
            current = current.parent
        return current
    
    @property
    def all_tops(self):
        tops = set()
        def climb(n):
            if n.parent is None or n.parent is n:
                tops.add(n)
            else:
                climb(n.parent)
        climb(self)
        return tops
    
    def is_top(self):
        """Return True if this node is a top-level node (no parent)."""
        return self.parent is None

    @property
    def children(self):
        """Return all child nodes."""
        return list(self.child_nodes.values())

    # -------------------------
    # Simple runtime helpers (existing behavior preserved)
    # -------------------------
    def _test_run(self, t: float = 1.0):
        """Run the environment for short time as a smoke test (used by validate/run_simulation)."""
        self.env.run(until=t)

    def validate_and_test_run(self, run_time: float = 1.0):
        """
        Combined check: validate structure (duplicates) and then perform a very short env.run to ensure nothing crashes.
        """
        val = self.validate(verbose=True)
        if not val["ok"]:
            raise RuntimeError("Validation failed due to duplicate IDs.")
        # run a short smoke-run; user subclasses should ensure their processes are well-behaved
        self._test_run(run_time)
        return True

    def run_simulation(self, time: float):
        """
        Validate and then run the simulation up to `time`. After run, attempt to call finalization hooks
        if provided on nodes/edges (for bookkeeping).
        """
        self.validate(verbose=True)
        self.env.run(until=time)

        nodes, edges = self.get_all_nodes_edges()
        for node in nodes.values():
            if hasattr(node, "update_final_state_time"):
                try:
                    node.update_final_state_time(time)
                except Exception:
                    raise Exception(f"Error while updating final state time of node '{node.id}'")
        for edge in edges.values():
            if hasattr(edge, "update_final_buffer_avg_content"):
                try:
                    edge.update_final_buffer_avg_content(time)
                except Exception:
                    raise Exception(f"Error while updating final buffer average content of edge '{edge.id}'")
