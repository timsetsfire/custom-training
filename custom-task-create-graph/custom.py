import pandas as pd
import dgl
import torch
from pathlib import Path
import pickle
import json

def dict_to_dgl(graph):
    e = graph["edges"] 
    u,v = list(zip(*e))
    g = dgl.graph((u,v))
    g.ndata["attr"] = torch.ones(g.num_nodes(), 1)
    return g

def load_model(code_dir):
    return "dummy"

def fit(X, y, output_dir, **kwargs):
    output_dir_path = Path(output_dir)
    if output_dir_path.exists() and output_dir_path.is_dir():
        with open("{}/dummy.pkl".format(output_dir), "wb") as fp:
            pickle.dump("dummy", fp)

def transform(data, transformer):
    """
    used to create a graph from the data.  at a minimum, expected each record of data 
    to be a string that parses to a dictionary with keys: edges, vertices
    """
    columns = data.columns
    if "graph" in columns:
        print("graph column available")
        parsed_data = df["graph"].map( lambda graph_as_string: json.loads(graph_as_string))
        dgl_graphs = map( lambda x: dict_to_dgl(x), parsed_data.values)
        dgl_graphs_bytes = map(lambda g: pickle.dumps(g), dgl_graphs)
        data["dgl_graph"] = list(dgl_graphs_bytes)
    else:
        print("no column named graph found")

    return data


