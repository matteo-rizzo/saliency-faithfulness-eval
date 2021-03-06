# Saliency-faithfulness-eval

A suite of tests to assess attention faithfulness for interpretability.

## Support

### Tests for Faithfulness Evaluation

| ID             | Code | What is tested?                                              | Reference |
| -------------- | ---- | ------------------------------------------------------------ | --------- |
| `acc`          | WP1  | Impact of attention on accuracy                              | [4]       |
| `mlp`          | WP2  | Impact on accuracy of using attention to guide non-contextual MLP model | [4]       |
| `adv`          | WP3  | Impact on predictions of adversarial attention distributions | [2, 4]    |
| `ers` (single) | SS1  | Impact on predictions of single weights erasure              | [3]       |
| `ers` (multi)  | SS2  | Impact on predictions groups of weights erasure              | [3]       |
| `rand_params`  | A1   | Impact on attention weights of model parameters randomization | [1]       |
| `rand_labels`  | A2   | Impact on attention weights of training labels randomization | [1]       |

1. Adebayo et al., *“Sanity Checks for Saliency Maps”*, NIPS (
   2018) [ [Paper](https://dl.acm.org/doi/10.5555/3327546.3327621) - [Code](https://github.com/adebayoj/sanity_checks_saliency) ]
2. Jain & Wallace, *“Attention is not Explanation”*, NAACL (
   2019) [ [Paper](https://www.aclweb.org/anthology/N19-1357/) - [Code](https://github.com/successar/AttentionExplanation) ]
3. Serrano & Smith, *“Is Attention Interpretable?”,* ACL (
   2019) [ [Paper](https://www.aclweb.org/anthology/P19-1282/) - [Code](https://github.com/serrano-s/attn-tests) ]
4. Wiegreffe & Pinter, *“Attention is not not Explanation”*, EMNLP (
   2019) [ [Paper](https://www.aclweb.org/anthology/D19-1002/) - [Code](https://github.com/sarahwie/attention) ]

### Tasks

* Color Constancy
* Temporal Color Constancy (TCC)

### Datasets

+ **TCC:** [TCC](https://github.com/yanlinqian/Temporal-Color-Constancy), presented in Qian et al. “A Benchmark for
  Temporal Color Constancy” ArXiv (
  2020) [[Paper](https://arxiv.org/abs/2003.03763) - [Code](https://github.com/yanlinqian/Temporal-Color-Constancy)].
  The dataset can be downloaded at www.cs.ubc.ca/research/tcc/

## Running the code

### Installation

The code has been developed and tested on Ubuntu 20.10 using Python3.9 and some functionalities introduced in torch 1.9.0. Please install the required Python packages listed in `requirements.txt`. Using a `virtualenv` is not compulsory but strongly advised.

### Configuring the dataset

Paths to dataset are relative to a base path which is configurable inside `auxiliary/settings.py` via the `PATH_TO_DATASET = path/to/dataset` folder. Please make sure the desired dataset is stored at that path under a folder named coherently with the paths used inside the user-defined PyTorch dataset class. For example, the TCC dataset implementation at `classes/tasks/ccc/multiframe/data/TCC.py` refer to a folder named `tcc`, that should be found at `path/to/dataset/color_checker`.

### Tests

All tests can be run from shell using the scripts (`.sh` files) into the `eval/tests` subdirectories. Each script allows for multiple configuration options that can be edited within the file and are thereby described.

#### WP1: Accuracy of learned vs uniform saliency

1. Run `python3 eval/tests/acc/acc_test.sh` to measure the impact on accuracy of using learned attention weights versus random attention weights

#### WP3: Training an adversarial saliency model

1. Run `python3 eval/tests/adv/variance_test.sh` to training the models with multiple different random seeds. This step is optional but strongly advised
2. Run `python3 eval/tests/adv/adv_test.sh` to train the adversarial attention models
3. Run `python3 eval/analysis/adv/adv_analysis.sh` to analyze the test output data

#### WP2: Saliency-guided MLP model

1. Run `python3 eval/tests/mlp/mlp_test.sh` to train the MLP models either guided by imposed attention weights, learning
   their own attention weights, or not using attention at all

#### SS1-SS2: Erasing saliency weights

1. Run `python3 eval/tests/ers/save_grads.sh` to save the attention gradients of each model at test time. These values
   will be used to ground the gradient-based criteria for the multi weights erasure
2. Run `python3 eval/tests/ers/erasure.sh` to run either the single or the multi weight erasure
3. Run `python3 eval/analysis/ers/ers_analysis.sh` to analyze the test output data

#### A1-A2: Parameters' and input labels' randomization

* Run `python3 eval/tests/ers/rand_params_test.sh` to run the parameters' randomization test
* Run `python3 eval/tests/ers/rand_labels_test.sh` to run the labels' randomization test
