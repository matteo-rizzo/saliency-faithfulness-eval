# faithful-attention-eval

A suite of tests to assess attention faithfulness for interpretability.

## Main references

* [Is Attention Interpretable?](https://www.aclweb.org/anthology/P19-1282/) by Serrano & Smith
* [Attention is not Explanation](https://www.aclweb.org/anthology/N19-1357/) by Jain & Wallace
* [Attention is not not Explanation](https://www.aclweb.org/anthology/D19-1002/) by Wiegreffe & Pinter

## Testing a new dataset

* Move your desired dataset to a folder under dataset
* Create a new PyTorch dataset as a new file at `classes/data`

## Testing a new model

* Create a file storing your model as a PyTorch `nn.Module` at `classes/modules`
* Create a new adversary model at `classes/adv`
* Move your base model, and the stored attention values and predictions, to a new folder under `trained_models`