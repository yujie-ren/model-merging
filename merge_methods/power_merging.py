############
# 1. How to implement power_merging?
# 1) Merging Equation
#   LLM_new = coeff*LLM + (1-coeff)*LLMinLLaVA
#   LLaVA_new = {LLM_new, MLP, CLIP}
# 2) Coefficient Setting Principle
#   Give different layers different coefficient, change ratio according to power function
#   LLM = {embeding_layer, transformer_layers, norm_layer, lm_head}

# 2.1) If coeff_decrease
#   coeff_embeding_layer=coeff_upper, coeff_norm_layer=coeff_lm_head=coeff_lower
#   For transformer layers:
#   coeff_first_layer=coeff_upper, coeff_last_layer=coeff_lower
#   coeff_middle_layer is a power function
#   power function: y = (coeff_lower - coeff_upper) * x^power + coeff_upper (a>0)

# 2.2) If coeff_increase
#   coeff_embeding_layer=coeff_lower, coeff_norm_layer=coeff_lm_head=coeff_upper
#   For transformer layers:
#   coeff_first_layer=coeff_lower, coeff_last_layer=coeff_upper
#   coeff_middle_layer is a power function
#   power function: y = (coeff_upper - coeff_lower) * x^power + coeff_lower (a>0)


import torch
from typing import Dict
from merge_methods.utils import merge_embedding
import re


@torch.no_grad()
def merge(
        LLM: torch.nn.Module,
        vocab_LLM: Dict[str, int],
        LLMinLLaVA: torch.nn.Module,
        vocab_LLMinLLaVA: Dict[str, int],
        coeff_upper: float,
        coeff_lower: float,
        power: float,
        mode: str
) -> torch.nn.Module:
    r"""
    Layer-wise merging, change coeff of each layer according to power function
    LLMinLLaVA_new = coeff*LLM + (1-coeff)*LLMinLLaVA

    Parameters:
        LLM (torch.nn.Module): LLM Module
        vocab_LLM (Dict[str, int]): Vocabulary of LLM, vocab_LLM = {token: token_id}
        LLMinLLaVA (torch.nn.Module): LLMinLLaVA
        vocab_LLMinLLaVA (Dict[str, int]): Vocabulary of LLMinLLaVA, vocab_LLMinLLaVA = {token: token_id}
        coeff_upper (float): Coefficient upper limit
        coeff_lower (float): Coefficient upper limit
        power (float): Power of power function
        mode (str): coeff_increase or coeff_decrease.
                    From bottom layer to top layer, the the coefficient is increasing or decreasing
    """
    if not 0 <= coeff_lower <= coeff_upper <= 1:
        raise ValueError("0 <= lower_coeff <= upper_coeff <= 1 must be observed")
    if mode not in ["coeff_increase", "coeff_decrease"]:
        raise ValueError("mode must be either 'coeff_increase' or 'coeff_decrease'")

    num_of_transformerLayers = LLM.config.num_hidden_layers  # num of transformer layers

    if mode == "coeff_decrease":
        coeff_embedding = coeff_upper
        coeff_norm = coeff_lm_head = coeff_lower
    elif mode == "coeff_increase":
        coeff_embedding = coeff_lower
        coeff_norm = coeff_lm_head = coeff_upper

    for (name_LLM, param_LLM), (name_LLMinLLaVA, param_LLMinLLaVA) in zip(LLM.named_parameters(), LLMinLLaVA.named_parameters()):
        assert name_LLM == name_LLMinLLaVA, f"Name doesn't match: {name_LLM} vs {name_LLMinLLaVA}"

        if name_LLM == "model.embed_tokens.weight":  # embedding layer
            param_merged = merge_embedding(param_LLM, param_LLMinLLaVA, vocab_LLM, vocab_LLMinLLaVA, coeff_embedding)
            param_LLMinLLaVA.data.copy_(param_merged)

        elif name_LLM == "lm_head.weight":  # lm_head
            param_merged = merge_embedding(param_LLM, param_LLMinLLaVA, vocab_LLM, vocab_LLMinLLaVA, coeff_lm_head)
            param_LLMinLLaVA.data.copy_(param_merged)

        elif name_LLM == "model.norm.weight":  # norm_layer
            param_merged = coeff_norm * param_LLM.data + (1 - coeff_norm) * param_LLMinLLaVA.data
            param_LLMinLLaVA.data.copy_(param_merged)

        elif "model.layers" in name_LLM:  # transformer layers
            match = re.search(r'layers\.(\d+)\.', name_LLM)
            layer_number = int(match.group(1))
            layer_index = layer_number / max(num_of_transformerLayers-1, 1)

            # calculate coefficient for current layer
            coeff = power_function(layer_index, coeff_upper, coeff_lower, power, mode)
            param_merged = coeff * param_LLM.data + (1 - coeff) * param_LLMinLLaVA.data
            param_LLMinLLaVA.data.copy_(param_merged)

    return LLMinLLaVA


def power_function(layer_index, coeff_upper, coeff_lower, power, mode):
    x = layer_index
    if mode == "coeff_decrease":
        y = (coeff_lower - coeff_upper) * (x ** power) + coeff_upper
    elif mode == "coeff_increase":
        y = (coeff_lower - coeff_upper) * ((1-x) ** power) + coeff_upper
    coeff = y

    return coeff