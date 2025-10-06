import torch


@torch.no_grad()
def merge_embedding(embedMatrix_LLM, embedMatrix_LLMinLLaVA, vocab_LLM, vocab_LLMinLLaVA, merge_coeff):
    r"""
    Objective: merge embedding layer of LLM & LLMinLLaVA, since their embedding layers have different dimensions.
    Procedure:
        1) If a token exists in both LLM & LLMinLLaVA, we compute the weighted average of their embeddings in LLMinLLaVA_new.
        2) If a token only exists in LLMinLLaVA, we use its embedding directly in LLMinLLaVA_new.
        3) If a token only exists in LLM, we ignore it.

    Parameters:
        embeddingMatrix_LLM (torch.Tensor): Parameter of LLM's embedding layer
        embeddingMatrix_LLMinLLaVA (torch.Tensor): Parameter of LLMinLLaVA's embedding layer
        vocab_LLM (dict{token: token_id}): Vocabulary of LLM
        vocab_LLMinLLaVA (dict{token: token_id}): Vocabulary of LLMinLLaVA
        merge_coeff (float): merge_coeff*LLM + (1-merge_coeff)*LLMinLLaVA
    Returns:
        embeddingMatrix_LLMinLLaVA_new (torch.Tensor): Parameter of LLMinLLaVA_new's embedding layer
    """

    embedMatrix_LLMinLLaVA_new = embedMatrix_LLMinLLaVA.detach().clone()

    # Find all tokens exist in both vocab_LLM & vocab_LLMinLLaVA, list() is important!
    common_tokens = list(set(vocab_LLM.keys()) & set(vocab_LLMinLLaVA.keys()))

    if len(common_tokens) == 0:  # Have no common tokens
        return embedMatrix_LLMinLLaVA_new

    # Get common token ids in LLMinLLaVA
    common_ids_LLMinLLaVA = [vocab_LLMinLLaVA[token] for token in common_tokens]
    common_ids_LLM = [vocab_LLM[token] for token in common_tokens]

    # Merging common embeddings
    embedMatrix_LLMinLLaVA_new[common_ids_LLMinLLaVA] = (
            merge_coeff * embedMatrix_LLM[common_ids_LLM] +
            (1 - merge_coeff) * embedMatrix_LLMinLLaVA_new[common_ids_LLMinLLaVA]
    )

    return embedMatrix_LLMinLLaVA_new



















    # ratio = merging_coefficient
    # embedMatrix_LLMinLLaVA_new = torch.nn.Parameter(torch.zeros_like(embedMatrix_LLMinLLaVA))
    #
    # id2token_LLMinLLaVA = {v: k for k, v in vocab_LLMinLLaVA.items()}
    #
    # for i in range(embedMatrix_LLMinLLaVA.size(0)):
    #     if i in vocab_LLMinLLaVA.values():  # i = token_id_LLMinLLaVA
    #         token = id2token_LLMinLLaVA[i]
    #         assert len(token) == 1, "token_id is not unique in vocabulary of LLMinLLaVA!"
    #         token = token[0]
    #
    #         if token in vocab_LLM:  # this token exists in both LLM & LLMinLLaVA
    #             token_id_LLM = vocab_LLM[token]
    #             embedMatrix_LLMinLLaVA_new[i] = (
    #                     ratio * embedMatrix_LLM[token_id_LLM] + (1 - ratio) * embedMatrix_LLMinLLaVA[i]
    #             )
    #         else:  # this token only exists in LLMinLLaVA
    #             embedMatrix_LLMinLLaVA_new[i] = embedMatrix_LLMinLLaVA[i]
    #
    #     else:  # This row of embeddMatrix doesn't belong to the embedding, it is just for GPU calculation.
    #         embedMatrix_LLMinLLaVA_new[i] = embedMatrix_LLMinLLaVA[i]
    #
    # return embedMatrix_LLMinLLaVA_new
