

for lr in 0.5 1.0 1.5 2.0
do
    for run in 0 1 2
    do
        sbatch --output="output/rnn_sgd${lr}_${run}.json" --error="output/rnn_sgd${lr}_${run}.err" job.sh rnn $lr 0 sgd

        sbatch --output="output/lstm_sgd${lr}_${run}.json" --error="output/lstm_sgd${lr}_${run}.err" job.sh lstm $lr 0 sgd

        sbatch --output="output/lstm_dropout0.1_sgd${lr}_${run}.json" --error="output/lstm_sgd${lr}_${run}.err" job.sh lstm $lr 0.1 sgd
        sbatch --output="output/lstm_dropout0.2_sgd${lr}_${run}.json" --error="output/lstm_sgd${lr}_${run}.err" job.sh lstm $lr 0.2 sgd
        sbatch --output="output/lstm_dropout0.5_sgd${lr}_${run}.json" --error="output/lstm_sgd${lr}_${run}.err" job.sh lstm $lr 0.5 sgd

        sbatch --output="output/lstm_dropout0.1_adam${lr}_${run}.json" --error="output/lstm_dropout_adam${lr}_${run}.err" job.sh lstm $lr 0.1 adam
        sbatch --output="output/lstm_dropout0.2_adam${lr}_${run}.json" --error="output/lstm_dropout_adam${lr}_${run}.err" job.sh lstm $lr 0.2 adam
        sbatch --output="output/lstm_dropout0.5_adam${lr}_${run}.json" --error="output/lstm_dropout_adam${lr}_${run}.err" job.sh lstm $lr 0.5 adam
    done
done