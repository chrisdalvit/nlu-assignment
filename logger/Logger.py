import json

class Logger:
    
    def __init__(self, env) -> None:
        self.data = {
            "args": vars(env.args),
            "epochs": [],
            "final_perplexity": None
        }
    
    def add_epoch_log(self, epoch, train_loss, eval_loss, ppl):
        self.data["epochs"].append({ 
            "epoch": epoch,
            "train_loss": train_loss, 
            "eval_loss": eval_loss, 
            "perplexity": ppl
        })
        
    def set_final_ppl(self, ppl):
        self.data["final_perplexity"] = ppl    
    
    def dumps(self):
        return json.dumps(self.data)
    