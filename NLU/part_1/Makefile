hid_size = 200
emb_size = 300
lr = 0.0001
clip = 5
emb_dropout = 0.0
out_dropout = 0.0

benchmark_all: benchmark_baseline

benchmark_baseline:
	@for lr in 0.5 1.0 2.0 5.0 10.0 20.0 30.0; do \
		sbatch --output="output/baseline$${lr}.json" --error="output/baseline$${lr}.err" --job-name="baseline" job.sh \
			--hid-size $(hid_size) \
			--emb-size $(emb_size) \
			--lr $$lr; \
			--clip $(clip) \
			--emb-dropout $(emb_dropout) \
			--out-dropout $(out_dropout) \
			--hid-dropout $(hid_dropout) \
	done 
	