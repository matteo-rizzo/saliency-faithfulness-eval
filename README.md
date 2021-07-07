# faithful-attention-eval

A suite of tests to assess attention faithfulness for interpretability.

> NOTE: THIS IS A WORK IN PROGRESS

## Resources

### Main references

#### Project Roadmap

Matteo's [slides](https://docs.google.com/presentation/d/1JuJhpu43QZfxYxAFgYqIl4LrXuQuOXWnYdOXPufeEjg/edit?usp=sharing).

#### Tests for Faithfulness Evaluation

* Adebayo et al., “Sanity Checks for Saliency Maps”, NIPS (2018) [[Paper](https://dl.acm.org/doi/10.5555/3327546.3327621) - [Code](https://github.com/adebayoj/sanity_checks_saliency)]
* Serrano & Smith, *“Is Attention Interpretable?”,* ACL (2019) [[Paper](https://www.aclweb.org/anthology/P19-1282/) - [Code](https://github.com/serrano-s/attn-tests)]
* Jain & Wallace, *“Attention is not Explanation”*, NAACL (2019) [[Paper](https://www.aclweb.org/anthology/N19-1357/) - [Code](https://github.com/successar/AttentionExplanation)] 
* Wiegreffe & Pinter, *“Attention is not not Explanation”*, EMNLP (2019) [[Paper](https://www.aclweb.org/anthology/D19-1002/) - [Code](https://github.com/sarahwie/attention)]

#### Color Constancy

* Hu et al., *“FC4: Fully Convolutional Color Constancy with Confidence-weighted Pooling”*, CVPR (
    2017) [[Paper](https://www.microsoft.com/en-us/research/publication/fully-convolutional-color-constancy-confidence-weighted-pooling/) - [Code](https://github.com/matteo-rizzo/fc4-pytorch)]
* Qian et al. “A Benchmark for Temporal Color Constancy” ArXiv (
    2020) [[Paper](https://arxiv.org/abs/2003.03763) - [Code](https://github.com/yanlinqian/Temporal-Color-Constancy)]

#### Datasets

+ [Color Checker](https://www2.cs.sfu.ca/~colour/data/shi_gehler/)
+ [TCC](https://github.com/yanlinqian/Temporal-Color-Constancy)

## Running the code

### Installation

The code has been developed and tested on Ubuntu 20.10 using Python3.9 and some funtionalities introduced in torch
1.9.0. Please install the required Python packages listed in `requirements.txt`. Using a `virtualenv` is not compulsory
but strongly advised.

### Configuring the dataset

Paths to dataset are relative to a base path which is configurable inside `auxiliary/settings.py` via
the `PATH_TO_DATASET = path/to/dataset` folder. Please make sure the desired dataset is stored at that path under a
folder name coherently with paths used inside the user-defined PyTorch dataset class. For example, the “Color Checker”
dataset implementation at `classes/data/datasets.ColorChecker.py` refer to a folder named `color_checker`, that should
be found at `path/to/dataset/color_checker`.

### Tests

#### JW1-WP3: Training an adversarial model

1. Under the same root folder (at, e.g., `path/to/model`), store the ground truth predictions and attention masks in
   directories named `att` and `pred` respectively
2. Either run the code with `python3 tests/adv/train_adv.py --path_to_base_model "path/to/model"` or edit
   the `PATH_TO_BASE_MODEL` global variable inside `tests/adv/train_adv.py`.

#### SS1-SS2: Erasing saliency weights

## Extending the code

## Testing a new dataset

* Move your desired dataset to a folder under dataset
* Create a new PyTorch dataset as a new file at `classes/data`

## Testing a new model

* Create a file storing your model as a PyTorch `nn.Module` at `classes/modules`
* Create a new adversary model at `classes/adv`
* Move your base model, and the stored attention values and predictions, to a new folder under `trained_models`
