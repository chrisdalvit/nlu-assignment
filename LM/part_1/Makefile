
benchmark_part_1: part_1_sgd part_1_adam

part_1_sgd: 
	@for lr in 0.5 1.0 1.5 2.0; do \
		# Run baseline; \
		sbatch --output="output/rnn_sgd$${lr}.json" --error="output/rnn_sgd$${lr}.err" job.sh rnn $$lr 0 sgd; \

		# Run raw LSTM; \
		sbatch --output="output/lstm_sgd$${lr}.json" --error="output/lstm_sgd$${lr}.err" job.sh lstm $$lr 0 sgd; \

		# Run LSTMs with dropout; \
		for dropout in 0.1 0.2 0.5; do \
			sbatch --output="output/lstm_dropout$${dropout}_adam$${lr}.json" --error="output/lstm_dropout$${dropout}_adam$${lr}.err" job.sh lstm $$lr $$dropout sgd; \
		done \
	done

part_1_adam:
	@for lr in 0.001 0.0005 0.0001 0.00005 ; do \
		for dropout in 0.1 0.2 0.5 ; do \
			sbatch --output="output/lstm_dropout$${dropout}_adam$${lr}.json" --error="output/lstm_dropout$${dropout}_adam$${lr}.err" job.sh lstm $$lr $$dropout adam; \
		done \
	done \
	