# Description LM - Part 1

In the first part the baseline RNN from Lab 4 is modified to improve performance. Incrementally the following modifications are added
- Replace RNN with a Long-Short Term Memory (LSTM) network
- Add two dropout layers
    - One after the embedding layer
    - One before the last linear layer
- Replace SGD with AdamW
The models are evaluated on the PennTreeBank using perplexity (PPL).

# Run the code
All computations were conducted on the Marzola cluster of the University of Trento. To run the code clone the repository onto the cluster. To run all evaluations of the first part do 
```bash
# Assuming you are in the root of the git repositroy
cd LM/part_1
mkdir output
make benchmark_all
```
This submits all benchmarking jobs to the compute nodes of the cluster. The results are stored in the ```output``` folder.

Alternativerly, you can run 
```bash
make benchmark_sgd
```
or 
```bash
make benchmark_adam
```
to only perfrom a subset of the benchmarking.

If you want to test a custom configuration adjust the parameters in the ```custom_job.sh``` and run
```bash
sbatch custom_job.sh
```
The results are stored in ```output/custom.json```.