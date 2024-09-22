from .LM_RNN import LM_RNN
from utils.utils import init_weights

AVAILABLE_MODEL_TYPES = ["rnn"]

def get_model(model_type, emb_size, hid_size, env):
    if model_type not in AVAILABLE_MODEL_TYPES
        return None
    elif model_type == "rnn":
        model = LM_RNN((emb_size, hid_size, len(env.lang), pad_index=env.pad_token_id).to(env.device))
    return model.apply(init_weights)