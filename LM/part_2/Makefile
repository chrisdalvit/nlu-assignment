emb_dropout = 0.4 
out_dropout = 0.1
hid_dropout = 0.25
num_layers = 2
lr = 20.0
epochs = 100
emb_size = 400
train_batch_size = 20

benchmark_all: benchmark_baseline benchmark_weight_tying benchmark_var_dropout benchmark_ntasgd

benchmark_baseline:
	@for lr in 0.5 1.0 2.0 5.0 10.0 20.0 30.0; do \
		sbatch --output="output/$${num_layers}lstm$${lr}.json" --error="output/$${num_layers}lstm$${lr}.err" --job-name="baseline2" job.sh \
			--optim sgd \
			--emb-dropout $(emb_dropout) \
			--out-dropout $(out_dropout) \
			--hid-dropout $(hid_dropout) \
			--num-layers $(num_layers) \
			--epochs $(epochs) \
			--emb-size $(emb_size) \
			--train-batch-size $(train_batch_size) \
			--lr $$lr; \
	done 

benchmark_weight_tying:
	@for lr in 0.5 1.0 2.0 5.0 10.0 20.0; do \
		sbatch --output="output/$${num_layers}lstm_wt$${lr}.json" --error="output/$${num_layers}lstm_wt$${lr}.err" --job-name="wt2" job.sh \
			--optim sgd \
			--weight-tying \
			--emb-dropout $(emb_dropout) \
			--out-dropout $(out_dropout) \
			--hid-dropout $(hid_dropout) \
			--num-layers $(num_layers) \
			--epochs $(epochs) \
			--emb-size $(emb_size) \
			--train-batch-size $(train_batch_size) \
			--lr $$lr; \
	done

benchmark_var_dropout:
	@for lr in 0.5 1.0 2.0 5.0 10.0 20.0; do \
		sbatch --output="output/$${num_layers}lstm_wt_vdrop$${lr}.json" --error="output/$${num_layers}lstm_wt_vdrop$${lr}.err" --job-name="vdrop2" job.sh \
			--optim sgd \
			--weight-tying \
			--variational-dropout \
			--emb-dropout $(emb_dropout) \
			--out-dropout $(out_dropout) \
			--hid-dropout $(hid_dropout) \
			--num-layers $(num_layers) \
			--epochs $(epochs) \
			--emb-size $(emb_size) \
			--train-batch-size $(train_batch_size) \
			--lr $$lr; \
	done 

benchmark_ntasgd:
	@for lr in 1.0 2.0 5.0 10.0 20.0; do \
		sbatch --output="output/$${num_layers}lstm_wt_vdrop_ntasgd$${lr}.json" --error="output/$${num_layers}lstm_wt_vdrop_ntasgd$${lr}.err" --job-name="ntasgd2" job.sh \
			--optim nt-avgsgd \
			--weight-tying \
			--variational-dropout \
			--emb-dropout $(emb_dropout) \
			--out-dropout $(out_dropout) \
			--hid-dropout $(hid_dropout) \
			--num-layers $(num_layers) \
			--epochs $(epochs) \
			--emb-size $(emb_size) \
			--train-batch-size $(train_batch_size) \
			--lr $$lr; \
	done
	