Code for paper - [Lexical-constrained-aware neural machine translation](https://www.ijcai.org/Proceedings/2020/496)

### Install and Data preprocess
The code is implemented on [fairseq v0.6.1](https://github.com/pytorch/fairseq/tree/v0.6.1), follow the same steps to install and prepare the processed fairseq dataset, the WMT process script is [here](https://github.com/pytorch/fairseq/blob/v0.6.1/examples/translation/prepare-wmt14en2de.sh).

### Run experiment
See scripts/run.sh

### Citation

```bibtex
@inproceedings{chen2020leca,
  title     = {Lexical-Constraint-Aware Neural Machine Translation via Data Augmentation},
  author    = {Chen, Guanhua and Chen, Yun and Wang, Yong and Li, Victor O.K.},
  booktitle = {Proceedings of {IJCAI} 2020: Main track},          
  pages     = {3587--3593},
  year      = {2020},
  month     = {7},
}
```
