SMALL_POSITIVE_CONST = 1e-4

def evaluate_ote(gold_ot, pred_ot, lang):
    """
    evaluate the model performce for the ote task
    :param gold_ot: gold standard ote tags
    :param pred_ot: predicted ote tags
    :return:
    """
    assert len(gold_ot) == len(pred_ot)
    n_samples = len(gold_ot)
    # number of true positive, gold standard, predicted opinion targets
    n_tp_ot, n_gold_ot, n_pred_ot = 0, 0, 0
    for i in range(n_samples):
        n_hit_ot = 0
        g_ot = gold_ot[i]
        p_ot = pred_ot[i]
        # hit number 
        for j, t in enumerate(p_ot):
            # count the number of correctly predicted opinion targets
            if t == g_ot[j] and lang.id2slot[t.item()].startswith("T"): 
                n_hit_ot += 1
        n_tp_ot += n_hit_ot
        # count the number of aspects in g_ot
        n_gold_ot += sum([1 for t in g_ot if lang.id2slot[t.item()].startswith("T")])
        # count the number of aspects in p_ot
        n_pred_ot += sum([1 for t in p_ot if lang.id2slot[t.item()].startswith("T")])
    # add 0.001 for smoothing
    # calculate precision, recall and f1 for ote task
    ot_precision = float(n_tp_ot) / float(n_pred_ot + SMALL_POSITIVE_CONST)
    ot_recall = float(n_tp_ot) / float(n_gold_ot + SMALL_POSITIVE_CONST)
    ot_f1 = 2 * ot_precision * ot_recall / (ot_precision + ot_recall + SMALL_POSITIVE_CONST)
    return {
        'precision': ot_precision,  
        'recall': ot_recall,
        'f1': ot_f1
    }