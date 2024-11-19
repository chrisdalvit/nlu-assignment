# Description NLU - Part 2
In the second part pre-trained BERT models are fine-tuned using a multi-task learning setting on intent classification and slot filling. The models are evaluated on the ATIS dataset using the accuracy for intent lassification and F1 score with conll for slot filling.

# Run the code
All computations were conducted on the Marzola cluster of the University of Trento. To run the code clone the repository onto the cluster. To run all evaluations of the first part do 
```bash
# Assuming you are in the root of the git repositroy
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd NLU/part_2
mkdir output
make benchmark_all
```
This submits all benchmarking jobs to the compute nodes of the cluster. The results are stored in the ```output``` folder.
Alternativerly, you can run 
```bash
make benchmark_base 
```
or 
```bash
make benchmark_large 
```
or 
```bash
make benchmark_base_dropout 
```
or 
```bash
make benchmark_large_dropout
```
or 
```bash
make benchmark_medium 
```
or 
```bash
make benchmark_mini 
```
or 
```bash
make benchmark_small 
```
or 
```bash
make benchmark_tiny 
```
to only perform a subset of the benchmarking.