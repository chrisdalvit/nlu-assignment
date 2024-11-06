hid_size = 200
emb_size = 300
adam_lrs = 0.0001 0.0005 0.001 0.005
sgd_lrs = 4 5 6 7
emb_dropout = 0.2
out_dropout = 0.2
hid_dropout = 0.2
num_layers = 2
train_batch_size = 64

benchmark_all: benchmark_baseline benchmark_bidirectional benchmark_dropout benchmark_sgd

benchmark_baseline:
	@for lr in $(adam_lrs); do \
		sbatch --output="output/baseline$${lr}.json" --error="output/baseline$${lr}.err" --job-name="baseline" job.sh \
			--name baseline \
			--hid-size $(hid_size) \
			--emb-size $(emb_size) \
			--lr $$lr \
			--num-layers $(num_layers) \
			--train-batch-size $(train_batch_size); \
	done

benchmark_bidirectional:
	@for lr in $(adam_lrs); do \
		sbatch --output="output/bidirectional$${lr}.json" --error="output/bidirectional$${lr}.err" --job-name="bidirectional" job.sh \
			--name bidirectional \
			--bidirectional \
			--hid-size $(hid_size) \
			--emb-size $(emb_size) \
			--lr $$lr \
			--num-layers $(num_layers) \
			--train-batch-size $(train_batch_size); \
	done

benchmark_dropout:
	@for lr in $(adam_lrs); do \
		sbatch --output="output/dropout$${lr}.json" --error="output/dropout$${lr}.err" --job-name="dropout" job.sh \
			--name dropout \
			--bidirectional \
			--hid-size $(hid_size) \
			--emb-size $(emb_size) \
			--lr $$lr \
			--num-layers $(num_layers) \
			--train-batch-size $(train_batch_size) \
			--emb-dropout $(emb_dropout) \
			--hid-dropout $(hid_dropout) \
			--out-dropout $(out_dropout); \
	done

benchmark_sgd:
	@for lr in $(sgd_lrs); do \
		sbatch --output="output/sgd$${lr}.json" --error="output/sgd$${lr}.err" --job-name="sgd" job.sh \
			--name sgd \
			--optim sgd \
			--bidirectional \
			--hid-size $(hid_size) \
			--emb-size $(emb_size) \
			--lr $$lr \
			--num-layers $(num_layers) \
			--train-batch-size $(train_batch_size) \
			--emb-dropout $(emb_dropout) \
			--hid-dropout $(hid_dropout) \
			--out-dropout $(out_dropout); \
	done
	