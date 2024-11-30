import torch 

def evaluate_ote(gold_ot, pred_ot, lang, epsilon=1e-4):
    """
    evaluate the model performce for the ote task
    :param gold_ot: gold standard ote tags
    :param pred_ot: predicted ote tags
    :return:
    """
    assert len(gold_ot) == len(pred_ot)
    n_tp_ot, n_gold_ot, n_pred_ot = 0, 0, 0
    n_tp_t = 0
    
    for g_ot, p_ot in zip(gold_ot, pred_ot):
        n_tp_ot += torch.sum((p_ot == g_ot) & torch.clone(p_ot).apply_(lang.is_target)) # are target and have same sentiment
        n_tp_t += torch.sum(torch.clone(g_ot).apply_(lang.is_target) & torch.clone(p_ot).apply_(lang.is_target)) # are target, ignore sentiment
        n_gold_ot += torch.clone(g_ot).apply_(lang.is_target).sum() # aspects in g_ot
        n_pred_ot += torch.clone(p_ot).apply_(lang.is_target).sum() # aspects in p_ot
    
    # calculate precision, recall and f1
    ot_precision = float(n_tp_ot) / float(n_pred_ot + epsilon)
    ot_recall = float(n_tp_ot) / float(n_gold_ot + epsilon)
    ot_f1 = 2 * ot_precision * ot_recall / (ot_precision + ot_recall + epsilon)
    
    target_p = float(n_tp_t) / float(n_pred_ot + epsilon)
    target_r = float(n_tp_t) / float(n_gold_ot + epsilon)
    target_f1 = 2 * target_p * target_r / (target_p + target_r + epsilon)
    return {
        'sentiment': {
            'precision': ot_precision,  
            'recall': ot_recall,
            'f1': ot_f1            
        },
        'target': {
            'precision': target_p,
            'recall': target_r,
            'f1': target_f1    
        }
    }