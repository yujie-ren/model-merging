###########
# 1 About weighted-averaging (model soup, linear) paper (https://arxiv.org/abs/2203.05482)
#   Model soup paper proposed 3 merging strategies：
#   ① Uniform Soup (simple averaging)
#       Directly average the parameters of all models, the coefficients of each model are the same.
#   ② Greedy Soup (selective averaging)
#       Try adding models to soup in order. If the performance of average-soup does not decrease
#       after adding a new model, keep the model.
#   ③ Learned Soup (weighted averaging)
#       How to get weights for each model/layer? (backpropagation)
#       Learn a coefficient for each model/layer based on a held-out validation set.


###########
# 2 How to implement weighted-averaging in our project?
#   LLM_new = λ*LLM + (1-λ)*LLMinLLaVA
#   LLaVA_new = {LLM_new, MLP, CLIP}


###########
# 3 How to merge "embedding layer" & "lm-head"? Since these two layers have different dimensions in LLM & LLMinLLaVA.
# (1) Solutions from other papers:
#   1) Embedding merging procedure from Mergekit (source: Transferring textual preferences paper)
#     ① If a token exists in the pre-trained model, we use its embedding from that model.
#     ② If a token appears in only one model (either LVLM or text-based RM), we use its embedding from that model.
#     ③ If a token appears in multiple models, we compute the average of its embeddings.
# (2) Solution for our project
#   ① If a token exists in both LLM & LLMinLLaVA, we compute the weighted average of its embedding.
#   ② If a token only exists in LLMinLLaVA, we use its embedding from LLMinLLaVA.


###########
# 4 How to select the coefficient λ (Lambda)?
# (1) Solutions from other papers
#   1) Model soup paper
#     Use backpropagation to solve a loss function on a held-out validation set

#   2) Transferring textual preferences paper (No public code)
#     Conduct a hyperparameter (coefficient) search, sample 400 instances from RLAIF-V training set as validation set.
#     Search for best λ within the range {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}

#   3) Safeguard paper (No public code)
#     Select coefficient λ based on validation set performance on the downstream tasks.
#     Determine λ by testing values in the set {0.2, 0.4, 0.6, 0.8}

#   4) DogeRM paper (with code)
#     Determining an appropriate weight factor λ depends on a small in-domain validation set.


# (2) Solution for our project
#   Coefficient search:
#     ① Construct a validation set. How ?????????
#           Question1: Can I directly select some samples from the test set as the validation set?
#           (No, we cannot use test set when building merging method)
#           Idea ?
#           Validation set = {0.3*MMMU}, Test set = {0.7*MMMU, MME-Perception, MME-Cognition}
#     ② Determine the range of coefficient search. λ = {0.2, 0.4, 0.6, 0.8}
#     ③ Run the merged model on the validation set with different coefficients.
#     ④ Select the coefficient that performs best on the validation set.


import torch
from merge_methods.utils import merge_embedding
from typing import Any, Dict  # for Type Hints


@torch.no_grad()
def merge(
        LLM: torch.nn.Module,
        vocab_LLM: Dict[str, int],
        LLMinLLaVA: torch.nn.Module,
        vocab_LLMinLLaVA: Dict[str, int],
        merge_coeff: float
) -> torch.nn.Module:
    r"""
    Merging LLM & LLMinLLaVA via weighted averaging (linear merging).
    LLMinLLaVA_new = merge_coeff*LLM + (1-merge_coeff)*LLMinLLaVA

    Parameters:
        LLM (torch.nn.Module): LLM model
        vocab_LLM (Dict[str, int]): Vocabulary of LLM, vocab_LLM = {token: token_id}
        LLMinLLaVA (torch.nn.Module): LLMinLLaVA
        vocab_LLMinLLaVA (Dict[str, int]): Vocabulary of LLMinLLaVA, vocab_LLMinLLaVA={token: token_id}
        merge_coeff (float): coefficient of merging (between 0 and 1)
    Returns:
        LLMinLLaVA (torch.nn.Module): merged LLM & LLMinLLaVA
    """
    if not 0 <= merge_coeff <= 1:
        raise ValueError("merge_coeff must be between 0 and 1")

    for (name_LLM, param_LLM), (name_LLMinLLaVA, param_LLMinLLaVA) in zip(LLM.named_parameters(), LLMinLLaVA.named_parameters()):
        assert name_LLM == name_LLMinLLaVA, f"Name doesn't match: {name_LLM} vs {name_LLMinLLaVA}"

        if name_LLM in ["model.embed_tokens.weight", "lm_head.weight"]:  # embedding layer or lm_head
            param_merged = merge_embedding(
                param_LLM.data, param_LLMinLLaVA.data, vocab_LLM, vocab_LLMinLLaVA, merge_coeff
            )  # param_LLM = embedMatrix_LLM, param_LLMinLLaVA = embedMatrix_LLMinLLaVA
            param_LLMinLLaVA.data.copy_(param_merged)

        elif (name_LLM == "model.norm.weight") or ("model.layers" in name_LLM):  # norm layer or transformer layers
            param_merged = merge_coeff * param_LLM.data + (1 - merge_coeff) * param_LLMinLLaVA.data
            param_LLMinLLaVA.data.copy_(param_merged)

    return LLMinLLaVA