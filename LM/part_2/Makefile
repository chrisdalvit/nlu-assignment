emb_dropout = 0.3 # 0.1
out_dropout = 0.3 # 0.1
hid_dropout = 0.5 # 0.25
num_layers = 3
lr = 5
epochs = 150
emb_size = 700

benchmark_all: benchmark_baseline benchmark_weight_tying benchmark_var_dropout benchmark_ntasgd

benchmark_baseline:
	sbatch --output="output/lstm.json" --error="output/lstm.err" --job-name="baseline2" job.sh \
		--optim sgd \
		--emb-dropout $(emb_dropout) \
		--out-dropout $(out_dropout) \
		--hid-dropout $(hid_dropout) \
		--num-layers $(num_layers) \
		--epochs $(epochs) \
		--emb-size $(emb_size) \
		--lr $(lr);

benchmark_weight_tying:
	sbatch --output="output/lstm_wt.json" --error="output/lstm_wt.err" --job-name="wt2" job.sh \
		--optim sgd \
		--weight-tying \
		--emb-dropout $(emb_dropout) \
		--out-dropout $(out_dropout) \
		--hid-dropout $(hid_dropout) \
		--num-layers $(num_layers) \
		--epochs $(epochs) \
		--emb-size $(emb_size) \
		--lr $(lr);

benchmark_var_dropout:
	sbatch --output="output/lstm_wt_vdrop.json" --error="output/lstm_wt_vdrop.err" --job-name="vdrop2" job.sh \
		--optim sgd \
		--weight-tying \
		--variational-dropout \
		--emb-dropout $(emb_dropout) \
		--out-dropout $(out_dropout) \
		--hid-dropout $(hid_dropout) \
		--num-layers $(num_layers) \
		--epochs $(epochs) \
		--emb-size $(emb_size) \
		--lr $(lr);

benchmark_ntasgd:
	sbatch --output="output/lstm_wt_vdrop_ntasgd.json" --error="output/lstm_wt_vdrop_ntasgd.err" --job-name="ntasgd2" job.sh \
		--optim nt-avgsgd \
		--weight-tying \
		--variational-dropout \
		--emb-dropout $(emb_dropout) \
		--out-dropout $(out_dropout) \
		--hid-dropout $(hid_dropout) \
		--num-layers $(num_layers) \
		--epochs $(epochs) \
		--emb-size $(emb_size) \
		--lr $(lr);
	