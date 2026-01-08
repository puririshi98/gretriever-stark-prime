from torch_geometric.llm.utils.backend_utils import (
   create_graph_from_triples,
   create_remote_backend_from_graph_data,
   make_pcst_filter,
   preprocess_triplet
)
from torch_geometric.llm.utils.feature_store import KNNRAGFeatureStore
from torch_geometric.llm.utils.graph_store import NeighborSamplingRAGGraphStore
from stark_qa import load_qa, load_skb
from torch.nn.utils import clip_grad_norm_
from torch_geometric.llm.models import SentenceTransformer, LLM, GRetriever
from torch_geometric.nn.models import SGFormer, GAT
from torch_geometric.data import Data
from torch_geometric.llm import RAGQueryLoader
from torch_geometric.loader import DataLoader
import os
import gc
from tqdm import tqdm
import random
import pandas as pd
import re
import argparse
# from g_retriever import (
#     adjust_learning_rate,
#     get_loss,
#     inference_step,
#     load_params_dict,
#     save_params_dict,
# )
parser = argparse.ArgumentParser(
   formatter_class=argparse.ArgumentDefaultsHelpFormatter, )
parser.add_argument('-e', '--epochs', type=int, default=3)
args = parser.parse_args()
###########
#helper funcs
"""This example provides helper functions for using the G-retriever model
(https://arxiv.org/abs/2402.07630) in PyG.


Requirements:
`pip install datasets transformers pcst_fast sentencepiece accelerate`




Example blog showing 2x accuracy over agentic graphRAG on real medical data
(integration with Neo4j Graph DB):
https://developer.nvidia.com/blog/boosting-qa-accuracy-with-graphrag-using-pyg-and-graph-databases/


https://github.com/neo4j-product-examples/neo4j-gnn-llm-example


See examples/llm/txt2kg_rag.py for e2e pipeline in PyG including:
- KG Creation
- Subgraph Retrieval
- GNN+LLM Finetuning
- Testing
- LLM Judge Eval


"""
import math


import torch
from torch import Tensor




def adjust_learning_rate(param_group: dict, LR: float, epoch: int,
                        num_epochs: int):
   """Decay learning rate with half-cycle cosine after warmup.


   Args:
       param_group (dict): Parameter group.
       LR (float): Learning rate.
       epoch (int): current epoch
       num_epochs (int): total epochs
   Returns:
       float: Adjusted learning rate.
   """
   min_lr = 5e-6
   warmup_epochs = 1
   if epoch < warmup_epochs:
       lr = LR
   else:
       lr = min_lr + (LR - min_lr) * 0.5 * (
           1.0 + math.cos(math.pi * (epoch - warmup_epochs) /
                          (num_epochs - warmup_epochs)))
   param_group['lr'] = lr
   return lr




def save_params_dict(model, save_path):
   """Saves a model's parameters, excluding non-trainable weights.


   Args:
       model (torch.nn.Module): The model to save parameters from.
       save_path (str): The path to save the parameters to.
   """
   # Get the model's state dictionary, which contains all its parameters
   state_dict = model.state_dict()


   # Create a dictionary mapping parameter names to their requires_grad status
   param_grad_dict = {
       k: v.requires_grad
       for (k, v) in model.named_parameters()
   }


   # Remove non-trainable parameters from the state dictionary
   for k in list(state_dict.keys()):
       if k in param_grad_dict.keys() and not param_grad_dict[k]:
           del state_dict[k]  # Delete parameters that do not require gradient


   # Save the filtered state dictionary to the specified path
   torch.save(state_dict, save_path)




def load_params_dict(model, save_path):
   # Load the saved model parameters from the specified file path
   state_dict = torch.load(save_path)


   # Update the model's parameters with the loaded state dictionary
   model.load_state_dict(state_dict)


   # Return the model with updated parameters
   return model




def get_loss(model, batch, model_save_name="gnn+llm") -> Tensor:
   """Compute the loss for a given model and batch of data.


   Args:
       model: The model to compute the loss for.
       batch: The batch of data to compute the loss for.
       model_save_name: The name of the model being used (e.g. 'llm').


   Returns:
       Tensor: The computed loss.
   """
   # Check the type of model being used to determine the input arguments
   if model_save_name == 'llm':
       # For LLM models
       return model(batch.question, batch.label, batch.desc)
   else:  # (GNN+LLM)
       return model(
           batch.question,  # ["list", "of", "questions", "here"]
           batch.x,  # [num_nodes, num_features]
           batch.edge_index,  # [2, num_edges]
           batch.batch,  # which node belongs to which batch index
           batch.label,  # list answers (labels)
           batch.edge_attr,  # edge attributes
           batch.desc  # list of text graph descriptions
       )




def inference_step(model, batch, model_save_name="gnn+llm",
                  max_out_tokens=128):
   """Performs inference on a given batch of data using the provided model.


   Args:
       model (nn.Module): The model to use for inference.
       batch: The batch of data to process.
       model_save_name (str): The name of the model (e.g. 'llm').
       max_out_tokens (int): The maximum number of tokens
           for our model to output.


   Returns:
       The output of the inference step.
   """
   # Check the type of model being used to determine the input arguments
   if model_save_name == 'llm':
       # Perform inference on the question and textual graph description
       return model.inference(batch.question, batch.desc,
                              max_out_tokens=max_out_tokens)
   else:  # (GNN+LLM)
       return model.inference(batch.question, batch.x, batch.edge_index,
                              batch.batch, batch.edge_attr, batch.desc,
                              max_out_tokens=max_out_tokens)
########
##### data prep
device='cuda'
ENCODER_MODEL_NAME_DEFAULT = "Qwen/Qwen3-Embedding-0.6B"
# 248 embed_batch_size default chosen to maximize speed on 80GB A100
def load_kg_in_pyg(dataset_name='prime', embed_batch_size=248):
   if not os.path.exists('saved_' + dataset_name + '_kg.pt'):
       skb = load_skb(dataset_name, download_processed=True, root=None)
       embedder = SentenceTransformer(
           model_name=ENCODER_MODEL_NAME_DEFAULT).to(device).eval()
       num_nodes = skb.num_nodes()
       node_strings = []
       for i in range(num_nodes):
           try:
               node_name = skb[i].dictionary['details']['name']
           except:
               node_name = skb[i].dictionary['name']
           if isinstance(node_name, list):
               node_name = '. '.join(node_name)
           try:
               node_summary = skb[i].dictionary['details']['summary']
               node_str = node_name + '. ' + node_summary
               node_strings.append(node_str.lower())
           except:
               node_strings.append(node_name.lower())


       edge_index = skb.edge_index
       edge_strings = []
       edge_string_index = skb.rel_type_lst()
       for edge_type in skb.edge_types:
           edge_strings.append(edge_string_index[edge_type].lower())
       # make triples
       triples = []
       for i in tqdm(range(len(edge_strings)), desc="Parsing raw triples"):
           triples.append(preprocess_triplet((node_strings[edge_index[0, i]], edge_strings[i], node_strings[edge_index[1, i]])))
       triples = list(dict.fromkeys(triples))
       data = create_graph_from_triples(
       triples=triples, embedding_model=embedder.encode,
       embedding_method_kwargs={
           "batch_size": embed_batch_size,
           "verbose": True
       }, pre_transform=preprocess_triplet)
       data.triples = triples
       del embedder
       torch.save(data, 'saved_' + dataset_name + '_kg.pt')
       gc.collect()
       torch.cuda.empty_cache()
   else:
       data = torch.load('saved_' + dataset_name + '_kg.pt', weights_only=False)
   return data


graph_data = load_kg_in_pyg()
triples = []
for t in graph_data.triples:
   triples.append(preprocess_triplet(t))
graph_data.triples = list(dict.fromkeys(triples))
print(graph_data)
def load_stark_prime_qa(dataset_name='prime'):
   qa_dataset = load_qa(dataset_name)
   skb = load_skb(dataset_name, download_processed=True, root=None)
   return [(q, '|'.join([skb[i].dictionary['name'].lower() for i in a])) for q, _, a, _ in qa_dataset]
LLM_GENERATOR_NAME_DEFAULT = "meta-llama/Meta-Llama-3.1-8B-Instruct"
graph_data.node_id = torch.arange(graph_data.x.size(0))
graph_data.edge_id = torch.arange(graph_data.edge_index.size(1))
# number of hops for neighborsampling
num_hops = 3
if os.path.exists("data_lists_for_finetune_gretrieve.pt"):
   data_lists = torch.load("data_lists_for_finetune_gretrieve.pt", weights_only=False)
else:
   qa_pairs = load_stark_prime_qa()
   text_embed = SentenceTransformer(
       model_name=ENCODER_MODEL_NAME_DEFAULT).to(device)
   subgraph_filter = make_pcst_filter(
           graph_data.triples,
           text_embed,
           topk=5,  # nodes
           topk_e=5,  # edges
           cost_e=.5,  # edge cost
           num_clusters=10)  # num clusters
   # number of neighbors for each seed node selected by KNN
   fanout = 10




   query_loader_config = {
       "k_nodes": 16,  # k for Graph KNN
       "num_neighbors": [fanout] * num_hops,  # number of sampled neighbors
       "encoder_model": text_embed,
   }
   fs, gs = create_remote_backend_from_graph_data(
       graph_data=graph_data, path="backend",
       graph_db=NeighborSamplingRAGGraphStore,
       feature_db=KNNRAGFeatureStore).load()


   query_loader = RAGQueryLoader(graph_data=(fs, gs),
                                   subgraph_filter=subgraph_filter,
                                   config=query_loader_config)
   total_data_list = []
   for q, a in tqdm(qa_pairs, desc="Building un-split dataset"):
       subgraph = query_loader.query(q)
       subgraph.label = a
       total_data_list.append(subgraph)
   random.shuffle(total_data_list)
   # 70:20:10 split
   data_lists = {}
   data_lists["train"] = total_data_list[:int(.7 * len(total_data_list))]
   data_lists["validation"] = total_data_list[int(.7 * len(total_data_list)
                                                   ):int(.9 *
                                                           len(total_data_list))]
   data_lists["test"] = total_data_list[int(.9 * len(total_data_list)):]
   torch.save(data_lists, "data_lists_for_finetune_gretrieve.pt")


#########
batch_size = 1
eval_batch_size = 2
train_loader = DataLoader(data_lists["train"], batch_size=batch_size,
                           drop_last=True, pin_memory=True, shuffle=True)
val_loader = DataLoader(data_lists["validation"],
                       batch_size=eval_batch_size, drop_last=False,
                       pin_memory=True, shuffle=False)
test_loader = DataLoader(data_lists["test"], batch_size=eval_batch_size,
                           drop_last=False, pin_memory=True, shuffle=False)


# # TODO combine node+edge into a model that fits into gretriever
# context_node_encoder = torch.load('context_node_encoder.pt')
# context_edge_encoder = torch.load('context_edge_encoder.pt')
# context_encoder = combine them
# use random init GAT to repro 32% first
context_encoder = GAT(
           in_channels=graph_data.x.size(1),
           hidden_channels=512,
           out_channels=512,
           heads=4,
           dropout=.5,
           num_layers=num_hops,
       ).to(device)
sys_prompt = (
   "You are an expert medical assistant that can answer "
   "any question from its knowledge, given a knowledge graph embedding. "
   "Just give the answer, without explanation.")
def train(train_loader, val_loader):
   epochs = args.epochs
   llm = LLM(model_name=LLM_GENERATOR_NAME_DEFAULT, sys_prompt=sys_prompt)


   model = GRetriever(llm=llm, gnn=context_encoder)


   params = [p for _, p in model.named_parameters() if p.requires_grad]
   lr = 1e-5
   optimizer = torch.optim.AdamW([{
       'params': params,
       'lr': lr,
       'weight_decay': 0.05
   }], betas=(0.9, 0.95))


   num_oom_errors = 0
   for epoch in range(epochs):
       model.train()
       epoch_loss = 0
       epoch_str = f'Epoch: {epoch + 1}|{epochs}'
       loader = tqdm(train_loader, desc=epoch_str)
       for step, batch in enumerate(loader):
           optimizer.zero_grad()
           # try removing this?
           batch.desc = ""
           try:
               loss = get_loss(model, batch)
               loss.backward()
               clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
               if (step + 1) % 2 == 0:
                   adjust_learning_rate(optimizer.param_groups[0], lr,
                                           step / len(train_loader) + epoch,
                                           epochs)
               optimizer.step()
               epoch_loss += float(loss.detach())
               if (step + 1) % 2 == 0:
                   lr = optimizer.param_groups[0]['lr']
           except torch.cuda.OutOfMemoryError:
               torch.cuda.empty_cache()
               print("Sequence length of last batch: ",
                       model.seq_length_stats[-1])
               # TODO: Implement CPU fallback (WIP)
               num_oom_errors += 1
       print("Sequence length stats: ")
       print("seq_len avg: ",
               sum(model.seq_length_stats) / len(model.seq_length_stats))
       print("seq_len min: ", min(model.seq_length_stats))
       print("seq_len max: ", max(model.seq_length_stats))
       print("Percent of OOM errors: ",
               num_oom_errors / len(train_loader))
       train_loss = epoch_loss / len(train_loader)
       print(epoch_str + f', Train Loss: {train_loss:4f}')


       # Eval Step
       val_loss = 0
       model.eval()
       with torch.no_grad():
           for batch in val_loader:
               batch.desc = ""
               loss = get_loss(model, batch)
               val_loss += loss.item()
           val_loss = val_loss / len(val_loader)
           print(epoch_str + f", Val Loss: {val_loss:4f}")
   torch.cuda.empty_cache()
   torch.cuda.reset_max_memory_allocated()
   model.eval()
   return model
model = train(train_loader, val_loader)
# define test using things from stark-qa
def compute_metrics(eval_output, skip_invalid_hit=True):
   df = pd.concat([pd.DataFrame(d) for d in eval_output])
   all_hit = []
   all_exact_hit_at_1 = []
   all_exact_hit_at_5 = []
   all_exact_hit_at_any = []
   all_precision = []
   all_recall = []
   all_recall_at_20 = []
   all_rr = []
   all_f1 = []
   all_num_preds = []
   for pred, label in zip(df.pred.tolist(), df.label.tolist()):
       pred = pred.split('[/s]')[0].strip().split('|')
       try:
           hit = re.findall(pred[0], label)
       except Exception as e:
           print(f'Label: {label}')
           print(f'Pred: {pred}')
           print(f'Exception: {e}')
           print('------------------')
           if skip_invalid_hit:
               continue
           else:
               hit = []


       all_hit.append(len(hit) > 0)


       label = label.split('|')
       exact_hit_at_1 = 1 * (pred[0] in label)
       exact_hit_at_5 = 1 * (len(set(pred[:5]).intersection(set(label))) > 0)
       matches = set(pred).intersection(set(label))
       matches_at_20 = len(set(pred[:20]).intersection(set(label)))
       precision = len(matches) / len(set(pred))
       recall = len(matches) / len(set(label))
       recall_at_20 = matches_at_20 / len(set(label))
       if recall + precision == 0:
           f1 = 0
       else:
           f1 = 2 * precision * recall / (precision + recall)


       for i, node in enumerate(pred):
           if node in label:
               rr = 1 / (i + 1)
               break
       else:
           rr = 0


       all_exact_hit_at_1.append(exact_hit_at_1)
       all_exact_hit_at_5.append(exact_hit_at_5)
       all_exact_hit_at_any.append(1 * (precision > 0))
       all_precision.append(precision)
       all_recall.append(recall)
       all_recall_at_20.append(recall_at_20)
       all_rr.append(rr)
       all_f1.append(f1)
       all_num_preds.append(len(pred))


   dataset_len = len(df.label.tolist())
   hit = sum(all_hit) / dataset_len
   exact_hit_at_1 = sum(all_exact_hit_at_1) / dataset_len
   exact_hit_at_5 = sum(all_exact_hit_at_5) / dataset_len
   exact_hit_at_any = sum(all_exact_hit_at_any) / dataset_len
   precision = sum(all_precision) / dataset_len
   recall = sum(all_recall) / dataset_len
   recall_at_20 = sum(all_recall_at_20) / dataset_len
   mrr = sum(all_rr) / dataset_len
   f1 = sum(all_f1) / dataset_len
   num_preds = sum(all_num_preds) / dataset_len


   print(f'F1:              {f1:.4f}')
   print(f'Precision:       {precision:.4f}')
   print(f'Recall:          {recall:.4f}')
   print(f'Substring hit@1: {hit:.4f}')
   print(f'Exact hit@1:     {exact_hit_at_1:.4f}')
   print(f'Exact hit@5:     {exact_hit_at_5:.4f}')
   print(f'Exact hit@any:   {exact_hit_at_any:.4f}')
   print(f'Recall@20:       {recall_at_20:.4f}')
   print(f'MRR:             {mrr:.4f}')
   print(f'Num predictions: {num_preds:.2f}')
eval_output = []
print("Final evaluation...")
for step, batch in enumerate(tqdm((test_loader))):
   batch.desc = ""
   with torch.no_grad():
       pred = inference_step(model, batch)
       eval_data = {
           'pred': pred,
           'question': batch.question,
           'label': batch.label
       }
       eval_output.append(eval_data)


compute_metrics(eval_output)





