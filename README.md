# synch: an extension of trasyn using gates from the Clifford hierarchy

A tensor network based approach to synthesise arbitrary single-qubit unitaries from 
extended universal gate sets. Concretely, here we provide the gate set composed of Clifford,
$T$ and $\sqrt{T}$. For details, check out our paper on [unfolded distillation](https://arxiv.org/abs/2604.15436)
and the original work on [tensor network based unitary synthesis](https://arxiv.org/abs/2503.15843). 

## Install
### Minimum Installation
From the root of the project install via pip
```bash
pip install -e .
```
For GPU acceleration use
```bash
pip install -e ".[cupy-cuda13]" # or [cupy-cuda11] depending on CUDA version
```
The CUDA Toolkit version can be found using `nvcc --version`.


## Usage
### Synthesize a single-qubit unitary
To synthesise a single qubit unitary using $T$ and $\sqrt{T}$ gates (using the gate sequences
provided in this package) we first need to set up a partitioner. This step is important to ensure
the synthesiser can sample from all sequences (and not just a subset). The `Partitioner` creates a 
a sequence of partitionings of the total cost allowance (if required) into smaller parts so that any 
possible sequence of $T$ and $\sqrt{T}$ (up to the cost allowance) is at least contained in one of 
the partitionings.

```Python console
from trasyn.src.synthesis import ErgodicPartitioner, Synthesiser
from trasyn.src.utils import random_unitary_2x2

partitioner = ErgodicPartitioner(
    max_partition_value=8,
    total_non_clifford_budget=10,
    costs={"t": 1.0, "q": 2.5},  # q represents sqrtT
)
synthesiser = Synthesiser(partitioner=partitioner)

target_unitary = random_unitary_2x2()
result = synthesiser.sample_and_synthesize(target_unitary)

error = result.error
sequence_str = result.seqstr
```
