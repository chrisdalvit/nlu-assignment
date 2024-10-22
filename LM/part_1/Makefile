
benchmark_all: benchmark_sgd benchmark_adam

benchmark_sgd: 
	@for lr in 0.5 1.0 1.5 2.0 ; do \
		sbatch --output="output/rnn_sgd$${lr}.json" --error="output/rnn_sgd$${lr}.err" job.sh --model rnn --lr $$lr --out-dropout 0 --emb-dropout 0 --optim sgd; \
		sbatch --output="output/lstm_sgd$${lr}.json" --error="output/lstm_sgd$${lr}.err" job.sh --model lstm --lr $$lr --out-dropout 0 --emb-dropout 0 --optim sgd; \
		for dropout in 0.1 0.2 0.5 ; do \
			sbatch --output="output/lstm_dropout$${dropout}_adam$${lr}.json" --error="output/lstm_dropout$${dropout}_sgd$${lr}.err" job.sh --model lstm --lr $$lr --out-dropout $$dropout --emb-dropout $$dropout --optim sgd; \
		done \
	done

benchmark_adam:
	@for lr in 0.001 0.0005 0.0001 0.00005 ; do \
		for dropout in 0.1 0.2 0.5 ; do \
			sbatch --output="output/lstm_dropout$${dropout}_adam$${lr}.json" --error="output/lstm_dropout$${dropout}_adam$${lr}.err" job.sh --model lstm --lr $$lr --out-dropout $$dropout --emb-dropout $$dropout --optim adam; \
		done \
	done \
	