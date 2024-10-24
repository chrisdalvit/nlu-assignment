hid_size = 200
emb_size = 300
lrs = 0.0001 0.0005 0.001
clip = 5
emb_dropout = 0.1
out_dropout = 0.1
hid_dropout = 0.1

benchmark_all: benchmark_baseline benchmark_bidirectional benchmark_dropout

benchmark_baseline:
	@for lr in $(lrs); do \
		sbatch --output="output/baseline$${lr}.json" --error="output/baseline$${lr}.err" --job-name="baseline" job.sh \
			--hid-size $(hid_size) \
			--emb-size $(emb_size) \
			--lr $$lr \
			--clip $(clip); \
	done

benchmark_bidirectional:
	@for lr in $(lrs); do \
		sbatch --output="output/bidirectional$${lr}.json" --error="output/bidirectional$${lr}.err" --job-name="bidirectional" job.sh \
			--bidirectional \
			--hid-size $(hid_size) \
			--emb-size $(emb_size) \
			--lr $$lr \
			--clip $(clip) \
			--emb-dropout $(emb_dropout) \
			--hid-dropout $(hid_dropout) \
			--out-dropout $(out_dropout); \
	done

benchmark_dropout:
	@for lr in $(lrs); do \
		sbatch --output="output/baseline$${lr}_edr$(emb_dropout)_odr$(out_dropout).json" --error="output/baseline$${lr}_edr$(emb_dropout)_odr$(out_dropout).err" --job-name="dropout" job.sh \
			--bidirectional \
			--hid-size $(hid_size) \
			--emb-size $(emb_size) \
			--lr $$lr \
			--clip $(clip) \
			--emb-dropout $(emb_dropout) \
			--hid-dropout $(hid_dropout) \
			--out-dropout $(out_dropout); \
	done
	