# Description LM - Part 1

In the second part following regularization techinques are applied to the LSTM baseline from the previous part

- Weight Tying
- Variational Dropout
- Non-monotonically Triggered AvSGD

The models are evaluated on the PennTreeBank using perplexity (PPL).

# Run the code
All computations were conducted on the Marzola cluster of the University of Trento. To run the code clone the repository on to the cluster. To run all evaluations of the second part do 
```bash
# Assuming you are in the root of the git repositroy
cd LM/part_2
mkdir output
make benchmark_all
```
This submits all benchmarking job to the compute nodes of the cluster. The results are stored in the ```output``` folder.

Alternativerly you can run 
```bash
make benchmark_baseline
```
or 
```bash
make benchmark_weight_tying
```
or 
```bash
make benchmark_var_dropout
```
or 
```bash
make benchmark_ntasgd
```
to only perfrom a subset of the benchmarking.

If you want to test a custom configuration adjust the parameters in the ```custom_job.sh``` and run
```bash
sbatch custom_job.sh
```
The results are stored in ```output/custom.json```.