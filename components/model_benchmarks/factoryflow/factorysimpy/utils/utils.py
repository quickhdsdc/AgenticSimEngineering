import random
from graphviz import Digraph

def get_edge_selector(sel_type, node, env, edge_type=None):
    """
    Returns a generator that yields selected indices from the node's edge list.

    Args:
    
        sel_type (str): The selection strategy. One of: 'RANDOM', 'ROUND_ROBIN'.
        node (object): The node object containing in_edges or out_edges.
        env (simpy.Environment): The simulation environment.
        edge_type (str, optional): Whether to select from 'out_edges' or 'in_edges'. Default is 'OUT'.

    Returns:

        generator: A generator yielding selected indices from the specified edge list.

    Raises:

        ValueError: If sel_type is not a valid selection strategy.
  
    """
    edge_type = edge_type.lower()
    assert edge_type in ["in", "out"], "edge_type must be either 'in' or 'out'."
    strategies = {
        "RANDOM": Random_edge_selector,
        
        "ROUND_ROBIN": RoundRobin_edge_selector,
       
    }

    if sel_type not in strategies:
        raise ValueError(
            f"Invalid selection type: {sel_type}. Must be one of: {', '.join(strategies.keys())}."
        )
    #print(edge_type)
    
    return strategies[sel_type](node, env, edge_type)

def Random_edge_selector(node, env, edge_type):
    while True:
    
        edges = getattr(node, f"{edge_type}_edges")
        yield random.randint(0, len(edges) - 1)

def RoundRobin_edge_selector(node, env, edge_type):
    i = 0
    while True:
        edges = getattr(node, f"{edge_type}_edges")
        yield i
        i = (i + 1) % len(edges)


def draw_blockdiagram1(node, output_file="factory_blockdiagram"):
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
        model_dict = node.extract_model_data()
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

        def add_cluster(node_id, params):
            print("!!!! adding cluster", params)
            print("!!!!!!")
            hierarchical_id= params.get("hierarchical_id", None)
            if hierarchical_id is None:
                raise ValueError(f"Cluster '{node_id}' missing 'hierarchical_id' in params.")
            if hierarchical_id in created_nodes:
                #print(f"Cluster {node_id} already created, skipping.")
                return

            with dot.subgraph(name=f"cluster_{hierarchical_id}") as c:
                c.attr(label=params["id"], style="rounded", color="gray40")
                created_nodes.add(hierarchical_id)
                print(f"\n\n{params}")

    
          

        # -------------------------
        # Add all model nodes
        # -------------------------
        for node_id, info in model_dict["nodes"].items():
            #print(node_id, info)

            if info["type"] not in ["Machine", "Source", "Sink", "Splitter", "Combiner"]:
                add_cluster(node_id, info["params"])
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


def draw_blockdiagram(self, output_file="factory_blockdiagram"):
    """
    Draw hierarchical Graphviz block diagram using runtime hierarchy.
    """

    # ------------------------------------------------------------
    # Ensure hierarchical IDs exist
    # ------------------------------------------------------------
    if not self.hierarchical_id:
        self.fill_hierarchical_id()

    # ------------------------------------------------------------
    # Extract model data ONCE
    # ------------------------------------------------------------
    model_dict = self.extract_model_data()

    node_params = {
        hid: info["params"]
        for hid, info in model_dict["nodes"].items()
    }
    edge_params = {
        hid: info["params"]
        for hid, info in model_dict["edges"].items()
    }

    dot = Digraph(
        comment="Factory Model Block Diagram",
        format="png",
    )
    dot.attr(
        rankdir="LR",
        fontsize="12",
        nodesep="0.5",
        ranksep="0.75",
        compound="true",
        size="16,14",
    )

    created = set()

    # ------------------------------------------------------------
    # Draw a rectangle (node or edge)
    # ------------------------------------------------------------
    def draw_box(g, hid, label, params, color):
        if hid in created:
            return

        tooltip = "\n".join(f"{k}: {v}" for k, v in params.items())
        g.node(
            hid,
            label=label,
            shape="box",
            style="filled",
            fillcolor=color,
            tooltip=tooltip,
        )
        created.add(hid)

    # ------------------------------------------------------------
    # Recursive hierarchy walk
    # ------------------------------------------------------------
    def walk(obj, g):
        has_children = bool(obj.child_nodes or obj.child_edges)

        # ---------------- Cluster ----------------
        if has_children:
            with g.subgraph(name=f"cluster_{obj.hierarchical_id}") as c:
                c.attr(label=obj.id, style="rounded", color="gray40")

                # child nodes
                for child in obj.child_nodes.values():
                    walk(child, c)

                # child edges (boxes)
                for edge in obj.child_edges.values():
                    hid = edge.hierarchical_id
                    draw_box(
                        c,
                        hid,
                        edge.id,
                        edge_params[hid],
                        color="lightyellow",
                    )

        # ---------------- Leaf node ----------------
        else:
            hid = obj.hierarchical_id
            draw_box(
                g,
                hid,
                obj.id,
                node_params[hid],
                color="palegreen",
            )

    # ------------------------------------------------------------
    # Build hierarchy
    # ------------------------------------------------------------
    walk(self, dot)

    # ------------------------------------------------------------
    # Draw connections: Node → Edge → Node
    # ------------------------------------------------------------
    for hid, p in edge_params.items():
        print(hid,p)
        assert "src_node" in p and "dest_node" in p, f"Edge '{hid}' in {p} missing src_node or dest_node."
        assert p["src_node"].hierarchical_id in created, f"Source node '{p['src_node'].hierarchical_id}' not created."
        assert p["dest_node"].hierarchical_id in created, f"Dest node '{p['dest_node'].hierarchical_id}' not created."
        
        src = p["src_node"].hierarchical_id
        dst = p["dest_node"].hierarchical_id
        print("no_issue-connecting", src, "to", dst)
        dot.edge(src, hid, color="gray40")
        dot.edge(hid, dst, color="gray40")

    return dot
