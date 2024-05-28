import os
import sys
import math
import pprint
from itertools import islice

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch_geometric as pyg
from torch import optim
from torch import nn
from torch.nn import functional as F
import pandas as pd
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ultra import tasks, util
from ultra.models import Ultra


from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc


separator = ">" * 30
line = "-" * 30

MatchingCVE = 'MatchingCVE'
MatchingCWE = 'MatchingCWE'

RELATIONS_TO_PREDICT = [MatchingCVE, MatchingCWE]

RELATION_TO_TARGET = {
    MatchingCVE: 'h',
    MatchingCWE: 't'
}


@torch.no_grad()
def test(cfg, model, test_data, cve_to_connected_cpes, all_cpes, device, logger):
    world_size = util.get_world_size()
    rank = util.get_rank()

    test_triplets = torch.cat([test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]).t()

    relation2loader = {}
    for relation in RELATIONS_TO_PREDICT:
        relation_id = test_data.rel_vocab[relation].item()
        relation_triplets = test_triplets[test_triplets[:, 2] == relation_id]
        sampler = torch_data.DistributedSampler(relation_triplets, world_size, rank)
        rel_testloader = torch_data.DataLoader(relation_triplets, cfg.train.batch_size, sampler=sampler)
        relation2loader[relation] = rel_testloader

    id2entity = {v.item(): k for k, v in test_data.entity_vocab.items()}

    relation = MatchingCVE
    test_loader = relation2loader[relation]

    tail2candidates = {}
    output_cves = ['cve-2023-4863', 'cve-2023-5217', 'cve-2023-38545']
    batch = torch.tensor([[0, test_data.entity_vocab[output_cve], test_data.rel_vocab[relation]] for output_cve in output_cves], 
                         device=device)

    t_batch, h_batch = tasks.all_negative(test_data, batch)
    t_pred = model(test_data, t_batch)
    h_pred = model(test_data, h_batch)

    # logger.warning(f'Running test on CVE: {cve}.')

    model.eval()
    _, h_batch = tasks.all_negative(test_data, batch)
    h_pred = model(test_data, h_batch)
    
    # create a dict from h_batch with tail as a key and empty list as value
    for i in range(h_batch.size(0)):
        tail = h_batch[i][0][1].item()
        tail_name = id2entity[tail]
        if tail in tail2candidates:
            # already processed this tail entity
            # irrelevant to process it again in case that we want to rank only one tail entity against all head entities
            continue

        # sort from highest to lowest score
        ids_sorted_by_rank = h_pred[i].argsort(descending=True).tolist()
        # get index of only those entities that are in all_cpes
        ids_sorted_by_rank = [e for e in ids_sorted_by_rank if e in all_cpes]

        entity_sorted_by_rank = [id2entity[e] for e in ids_sorted_by_rank]
        scores_sorted_by_rank = h_pred[i][ids_sorted_by_rank].tolist()
        is_connected_cpe = [e in cve_to_connected_cpes[tail] for e in ids_sorted_by_rank]
        tail2candidates[tail_name] = {'entity_id': entity_sorted_by_rank, 'score': scores_sorted_by_rank, 'is_connected_cpe': is_connected_cpe}

    for output_cve in output_cves:
        logger.warning(f'Output CVE: {output_cve}')
        # logger.warning(f'Tail entity candidates: {tail2candidates[output_cve]}')

        pd.DataFrame(tail2candidates[output_cve]).to_csv(f'./{output_cve}.csv', index=False)


def get_cve_to_connected_cpes(train_data, valid_data, test_data):
    cve_to_connected_cpes = {}
    rel_id = train_data.rel_vocab[MatchingCVE].item()
    for data in [train_data, valid_data, test_data]:
        data = torch.cat([data.target_edge_index, data.target_edge_type.unsqueeze(0)]).t()
        relevant_triples = data[data[:, 2] == rel_id]
        for triple in relevant_triples:
            cpe = triple[0].item()
            cve = triple[1].item()
            if cve in cve_to_connected_cpes:
                cve_to_connected_cpes[cve].add(cpe)
            else:
                cve_to_connected_cpes[cve] = {cpe}

    all_cpes = set()
    for connected_cpes in cve_to_connected_cpes.values():
        all_cpes.update(connected_cpes)

    return cve_to_connected_cpes, all_cpes


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + util.get_rank())

    logger = util.get_root_logger(file=False)
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
    
    task_name = cfg.task["name"]
    dataset = util.build_dataset(cfg)
    device = util.get_device(cfg)
    
    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)

    model = Ultra(
        rel_model_cfg=cfg.model.relation_model,
        entity_model_cfg=cfg.model.entity_model,
        # lm_vectors=util.load_language_model_vectors(train_data.entity_vocab, cfg.train.lm_vectors) if cfg.train.lm_vectors else None
    )


    if model.entity_model.lm_vectors is not None:
        shape = model.entity_model.lm_vectors.weight.data.shape
        logger.warning(f'Loaded {shape[0]} language model vectors, each of shape {shape[1:]}')

    if "checkpoint" in cfg and cfg.checkpoint is not None:
        state = torch.load(cfg.checkpoint, map_location="cpu")
        try:
            # ignore lm_vectors
            state["model"].pop("entity_model.lm_vectors.weight")
        except KeyError:
            logger.warning("No language model vectors in checkpoint")
        model.load_state_dict(state["model"], strict=False)

    #model = pyg.compile(model, dynamic=True)
    model = model.to(device)
    
    if task_name == "InductiveInference":
        # filtering for inductive datasets
        # Grail, MTDEA, HM datasets have validation sets based off the training graph
        # ILPC, Ingram have validation sets from the inference graph
        # filtering dataset should contain all true edges (base graph + (valid) + test) 
        if "ILPC" in cfg.dataset['class'] or "Ingram" in cfg.dataset['class']:
            # add inference, valid, test as the validation and test filtering graphs
            full_inference_edges = torch.cat([valid_data.edge_index, valid_data.target_edge_index, test_data.target_edge_index], dim=1)
            full_inference_etypes = torch.cat([valid_data.edge_type, valid_data.target_edge_type, test_data.target_edge_type])
            test_filtered_data = Data(edge_index=full_inference_edges, edge_type=full_inference_etypes, num_nodes=test_data.num_nodes)
            val_filtered_data = test_filtered_data
        else:
            # test filtering graph: inference edges + test edges
            full_inference_edges = torch.cat([test_data.edge_index, test_data.target_edge_index], dim=1)
            full_inference_etypes = torch.cat([test_data.edge_type, test_data.target_edge_type])
            test_filtered_data = Data(edge_index=full_inference_edges, edge_type=full_inference_etypes, num_nodes=test_data.num_nodes)

            # validation filtering graph: train edges + validation edges
            val_filtered_data = Data(
                edge_index=torch.cat([train_data.edge_index, valid_data.target_edge_index], dim=1),
                edge_type=torch.cat([train_data.edge_type, valid_data.target_edge_type])
            )
    else:
        # for transductive setting, use the whole graph for filtered ranking
        filtered_data = Data(edge_index=dataset._data.target_edge_index, edge_type=dataset._data.target_edge_type, num_nodes=dataset[0].num_nodes)
        val_filtered_data = test_filtered_data = filtered_data
    
    val_filtered_data = val_filtered_data.to(device)
    
    cve_to_connected_cpes, all_cpes = get_cve_to_connected_cpes(train_data, valid_data, test_data)
        

    test(cfg, model, test_data, cve_to_connected_cpes, all_cpes, device=device, logger=logger)
    # test_first_ground_truth(cfg, model, test_data, cve_to_connected_cpes, all_cpes, device=device, logger=logger)
