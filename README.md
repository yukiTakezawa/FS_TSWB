# FS-TSWB

This is the demo code for the [Fixed-Support Tree-Sliced Wasserstein Barycenter](https://arxiv.org/abs/2109.03431).

## Requirements
- Python (3.7.4)
- POT (0.7.0)
- scipy (1.3.1)
- numpy (1.19.5)
- treelib (1.6.1)
- sklearn (0.21.3)
- pytorch (1.7.0)
- mat4py (0.4.3)
- gensim (3.8.0)
- nltk (3.4.5)
- tqdm

## Quick Start
The sample code is contained in examples.ipynb.

### Details Experiments
You can reproduce the experimental results by running the following commands.
```
mkdir exp
bash mnist_time_support.sh
bash mnist_time_sample.sh
bash mnist_memory_support.sh
bash mnist_memory_sample.sh
```
`mnist_memory_support.sh` and `mnist_memory_sample.sh` is only available on Linux.

## Citation
```
@inproceedings{takezawa2022fixed,
    title = {Fixed Support Tree-Sliced Wasserstein Barycenter},
    author = {Yuki Takezawa and Ryoma Sato and Zornitsa Kozareva and Sujith Ravi and Makoto Yamada},
    booktitle = {International Conference on Artificial Intelligence and Statistics},
    year = {2022}
}
```
