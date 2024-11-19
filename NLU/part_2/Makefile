adam_lrs = 0.0001 0.00005 0.00001 
dropout = 0.2
epochs = 10

benchmark_all: benchmark_base benchmark_large benchmark_base_dropout benchmark_large_dropout benchmark_medium benchmark_mini benchmark_small benchmark_tiny

benchmark_base:
	@for lr in $(adam_lrs); do \
		sbatch --output="output/base$${lr}.json" --error="output/base$${lr}.err" --job-name="base" job.sh \
			--name base \
			--lr $$lr \
			--num-epochs $(epochs) \
			--bert-version bert-base-uncased; \
	done

benchmark_large:
	@for lr in $(adam_lrs); do \
		sbatch --output="output/large$${lr}.json" --error="output/large$${lr}.err" --job-name="large" job.sh \
			--name large \
			--lr $$lr \
			--num-epochs $(epochs) \
			--bert-version bert-large-uncased; \
	done

benchmark_base_dropout:
	@for lr in $(adam_lrs); do \
		sbatch --output="output/base$${lr}_dropout$${dropout}.json" --error="output/base$${lr}_dropout$${dropout}.err" --job-name="base_dropout" job.sh \
			--name base-dropout \
			--lr $$lr \
			--dropout $(dropout) \
			--num-epochs $(epochs) \
			--bert-version bert-base-uncased; \
	done

benchmark_large_dropout:
	@for lr in $(adam_lrs); do \
		sbatch --output="output/large$${lr}_dropout$(dropout).json" --error="output/large$${lr}_dropout$(dropout).err" --job-name="large_dropout" job.sh \
			--name large-droput \
			--lr $$lr \
			--dropout $(dropout) \
			--num-epochs $(epochs) \
			--bert-version bert-large-uncased; \
	done

benchmark_tiny:
	@for lr in $(adam_lrs); do \
		sbatch --output="output/tiny$${lr}_dropout$(dropout).json" --error="output/tiny$${lr}_dropout$(dropout).err" --job-name="tiny" job.sh \
			--name tiny \
			--lr $$lr \
			--dropout $(dropout) \
			--num-epochs $(epochs) \
			--bert-version bert-tiny-uncased; \
	done

benchmark_mini:
	@for lr in $(adam_lrs); do \
		sbatch --output="output/mini$${lr}_dropout$(dropout).json" --error="output/mini$${lr}_dropout$(dropout).err" --job-name="min" job.sh \
			--name mini \
			--lr $$lr \
			--dropout $(dropout) \
			--num-epochs $(epochs) \
			--bert-version bert-mini-uncased; \
	done

benchmark_small:
	@for lr in $(adam_lrs); do \
		sbatch --output="output/small$${lr}_dropout$(dropout).json" --error="output/small$${lr}_dropout$(dropout).err" --job-name="small" job.sh \
			--name small \
			--lr $$lr \
			--dropout $(dropout) \
			--num-epochs $(epochs) \
			--bert-version bert-small-uncased; \
	done

benchmark_medium:
	@for lr in $(adam_lrs); do \
		sbatch --output="output/medium$${lr}_dropout$(dropout).json" --error="output/medium$${lr}_dropout$(dropout).err" --job-name="medium" job.sh \
			--name medium \
			--lr $$lr \
			--dropout $(dropout) \
			--num-epochs $(epochs) \
			--bert-version bert-medium-uncased; \
	done