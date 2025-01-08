# Description NLU - Part 1

In the first part following modifications are applied to the LSTM baseline ```ModelIAS``` to improve performance

- Adding bidirectionality
- Adding dropout layer

The models are evaluated on the ATIS dataset using the accuracy for intent lassification and F1 score with conll for slot filling.

# Run the code
All computations were conducted on the Marzola cluster of the University of Trento. To run the code clone the repository onto the cluster. To run all evaluations of the first part do 
```bash
# Assuming you are in the root of the git repositroy
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd NLU/part_1
mkdir output
make benchmark_all
```
This submits all benchmarking jobs to the compute nodes of the cluster. The results are stored in the ```output``` folder.

Alternativerly, you can run 
```bash
make benchmark_baseline
```
or 
```bash
make benchmark_bidirectional 
```
or 
```bash
make benchmark_dropout 
```
or 
```bash
make benchmark_sgd
```
to only perform a subset of the benchmarking.

If you want to test a custom configuration adjust the parameters in the ```custom_job.sh``` and run
```bash
sbatch custom_job.sh
```
The results are stored in ```output/custom.json```.