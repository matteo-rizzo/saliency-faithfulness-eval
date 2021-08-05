# faithful-attention-eval

A suite of tests to assess attention faithfulness for interpretability.

> NOTE: THIS IS A WORK IN PROGRESS

## Resources

### Main references

#### Tests for Faithfulness Evaluation

* Adebayo et al., “Sanity Checks for Saliency Maps”, NIPS (
    2018) [[Paper](https://dl.acm.org/doi/10.5555/3327546.3327621) - [Code](https://github.com/adebayoj/sanity_checks_saliency)]
* Serrano & Smith, *“Is Attention Interpretable?”,* ACL (
    2019) [[Paper](https://www.aclweb.org/anthology/P19-1282/) - [Code](https://github.com/serrano-s/attn-tests)]
* Jain & Wallace, *“Attention is not Explanation”*, NAACL (2019) [[Paper](https://www.aclweb.org/anthology/N19-1357/)
    - [Code](https://github.com/successar/AttentionExplanation)] 
* Wiegreffe & Pinter, *“Attention is not not Explanation”*, EMNLP (
    2019) [[Paper](https://www.aclweb.org/anthology/D19-1002/) - [Code](https://github.com/sarahwie/attention)]

#### Color Constancy

* Hu et al., *“FC4: Fully Convolutional Color Constancy with Confidence-weighted Pooling”*, CVPR (
    2017) [[Paper](https://www.microsoft.com/en-us/research/publication/fully-convolutional-color-constancy-confidence-weighted-pooling/) - [Code](https://github.com/matteo-rizzo/fc4-pytorch)]
* Qian et al. “A Benchmark for Temporal Color Constancy” ArXiv (
    2020) [[Paper](https://arxiv.org/abs/2003.03763) - [Code](https://github.com/yanlinqian/Temporal-Color-Constancy)]

### Datasets

+ [Color Checker](https://www2.cs.sfu.ca/~colour/data/shi_gehler/)
+ [TCC](https://github.com/yanlinqian/Temporal-Color-Constancy)

## Running the code

### Installation

The code has been developed and tested on Ubuntu 20.10 using Python3.9 and some functionalities introduced in torch
1.9.0. Please install the required Python packages listed in `requirements.txt`. Using a `virtualenv` is not compulsory
but strongly advised.

### Configuring the dataset

Paths to dataset are relative to a base path which is configurable inside `auxiliary/settings.py` via
the `PATH_TO_DATASET = path/to/dataset` folder. Please make sure the desired dataset is stored at that path under a
folder name coherently with paths used inside the user-defined PyTorch dataset class. For example, the “Color Checker”
dataset implementation at `classes/data/datasets.ColorChecker.py` refer to a folder named `color_checker`, that should
be found at `path/to/dataset/color_checker`.

### Tests

All tests can be run from shell using the scripts (`.sh` files) into the `tests` folder. Each script allows for multiple
configuration options that can be edited within the file and are thereby described.

#### JW1-WP3: Training an adversarial model

* Run `python3 tests/adv/variance.sh` to training the models with multiple different random seeds. This step is optional
  but strongly advised
* Run `python3 tests/adv/adv.sh` to train the adversarial attention models

#### SS1-SS2: Erasing saliency weights

* Run `python3 tests/erasure/save_grads.sh` to save the attention gradients of each model at test time. These values
  will be used to ground the gradient-based criteria for the multi weights erasure
* Run `python3 tests/erasure/erasure.sh` to run either the single or the multi weight erasure

#### WP2: Attention-guided MLP

* Run `python3 tests/mlp/mlp.sh` to train the MLP models either guided by imposed attention weights or learning their
  own attention weights