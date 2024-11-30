# Description SA - Part 1
In the SA project a pre-trained Language model (such as BERT or RoBERTa) is fine-tuned for the Aspect Based Sentiment Analysis task.

The models are evaluated on the Laptop partition of SemEval2014 task 4 dataset using the precision, recall and F1 score.

# Run the code
All computations were conducted on the Marzola cluster of the University of Trento. To run the code clone the repository onto the cluster. To run all evaluations of the first part do 
```bash
# Assuming you are in the root of the git repositroy
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd SA/part_1
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
to only perform a subset of the benchmarking.