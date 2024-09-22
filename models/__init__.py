from .LM_RNN import LM_RNN
from .LM_LSTM import LM_LSTM
from utils.utils import init_weights

AVAILABLE_MODEL_TYPES = ["rnn", "lstm"]

def get_model(env):
    model_name = env.args.model
    if model_name not in AVAILABLE_MODEL_TYPES:
        return None
    if model_name == "rnn":
        model = LM_RNN(env.args.emb_size, env.args.hid_size, len(env.lang), pad_index=env.pad_token_id).to(env.device)
    elif model_name == "lstm":
        model = LM_LSTM(env.args.emb_size, env.args.hid_size, len(env.lang), pad_index=env.pad_token_id).to(env.device)
    return model.apply(init_weights)

def save_model():
    #
    # TODO: implement save function
    # 
    #path = "./output/model.pt"
    #torch.save(model.state_dict(), path)
    #model = LM_RNN(emb_size, hid_size, len(lang), pad_index=pad_token_id).to(device)
    #model.load_state_dict(torch.load(path))
    ...