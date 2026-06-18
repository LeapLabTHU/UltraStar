# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ELECTRA https://github.com/google-research/electra
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import json
import logging

def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75, text_lr_mult=1.0):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}
    if hasattr(model, 'encoder'):
        num_layers = len(model.encoder.blocks) + 1
    else:
        num_layers = len(model.blocks) + 1
    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)
        this_scale = layer_scales[layer_id]

        if group_name not in param_group_names:
            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
                "params_name": []
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
                "params_name": []
            }

        param_groups[group_name]["params"].append(p)
        param_groups[group_name]["params_name"].append(n)

    return list(param_groups.values())


def param_groups_lrd_encoder_and_predictor(encoder, predictor, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75, text_lr_mult=1.0):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    encoder_num_layers = len(encoder.blocks) + 2
    predictor_num_layers = len(predictor.predictor_blocks) + 2
    total_layers = encoder_num_layers + predictor_num_layers

    layer_scales = list(layer_decay ** (total_layers - 1 - i) for i in range(total_layers))

    for n, p in encoder.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, encoder_num_layers - 1)
        group_name = "layer_%d_%s" % (layer_id, g_decay)
        this_scale = layer_scales[layer_id]
        print(layer_id, n, group_name, this_scale)

        if group_name not in param_group_names:
            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
                "params_name": []
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
                "params_name": []
            }

        param_groups[group_name]["params"].append(p)
        param_groups[group_name]["params_name"].append(n)

    for n, p in predictor.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_predictor(n, encoder_num_layers, total_layers - 1)
        group_name = "layer_%d_%s" % (layer_id, g_decay)
        this_scale = layer_scales[layer_id]
        print(layer_id, n, group_name, this_scale)

        if group_name not in param_group_names:
            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
                "params_name": []
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
                "params_name": []
            }

        param_groups[group_name]["params"].append(p)
        param_groups[group_name]["params_name"].append(n)

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    name = name.replace('feature_model.', '')
    # if name in ['cls_token', 'pos_embed']:
    if 'cls_token' in name or 'pos_embed' in name:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers


def get_layer_id_for_predictor(name, start_layers, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    name = name.replace('feature_model.', '')
    # if 'timestep_embed' in name or 'pos_embed' in name or 'image_mask_token' in name or :
    if name in ['timestep_embed', 'pos_embed', 'image_mask_token', 'action_mask_token']:
        return start_layers
    elif name.startswith('image_feat_embed') or name.startswith('action_feat_embed'):
        return start_layers
    elif name.startswith('predictor_blocks'):
        return start_layers + int(name.split('.')[1]) + 1
    else:
        return num_layers

def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or 'bn' in name or name.endswith(".bias") or name in skip_list:
            # print(name)
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]