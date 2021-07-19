import pandas as pd
import pickle
import dgl 
import torch
from gin import GIN

def init(code_dir):

def load_model(code_dir):
   # Returning a string with value "dummy" as the model.
    model = GIN(2, 2, 1, 20, 2, 0, 0.01, "sum", "sum")
    model.load_state_dict(torch.load(os.path.join(input_dir, "gin_model.h5")))
    return model

def fit(X, y, output_dir, **kwargs):

    model = GIN(2, 2, 1, 20, 2, 0, 0.01, "sum", "sum")
    dgl_graphs = df2["dgl_graph"].values
    dgl_graphs = list( map ( lambda x: pickle.loads(x), dgl_graphs))
    batched_graph = dgl.batch(dgl_graphs)
    batched_labels = torch.tensor(y.values)

    opt = torch.optim.Adam(model.parameters(),lr=0.01)

    for epoch in range(500):
        for graphs, labels in zip(batched_graph, batched_labels):
            feats = graphs.ndata['attr'].float()
            logits = model(graphs, feats)
            loss = F.cross_entropy(logits, label)
            # print(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 100 == 0:
            print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))

    output_dir_path = Path(output_dir)
    if output_dir_path.exists() and output_dir_path.is_dir():
        torch.save(model.state_dict(), "{}/gin_model.h5".format(output_dir))

def score(data, model):

