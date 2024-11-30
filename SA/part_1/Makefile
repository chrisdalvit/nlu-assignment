adam_lrs = 0.0001 0.00005 0.00001 
epochs = 10

benchmark_all: benchmark_base benchmark_large

benchmark_base:
	@for lr in $(adam_lrs); do \
		sbatch --output="output/base$${lr}.json" --error="output/base$${lr}.err" --job-name="base" job.sh \
			--name base \
			--lr $$lr \
			--bert-version bert-base-uncased; \
	done

benchmark_large:
	@for lr in $(adam_lrs); do \
		sbatch --output="output/large$${lr}.json" --error="output/large$${lr}.err" --job-name="large" job.sh \
			--name large \
			--lr $$lr \
			--bert-version bert-large-uncased; \
	done
