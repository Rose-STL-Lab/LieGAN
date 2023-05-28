
This folder contains top tagging dataset loading modules. This code was adapted from [[1]](https://github.com/fizisist/LorentzGroupNetwork/tree/master/src/lgn/data) and [[2]](https://github.com/vgsatorras/egnn/tree/main/qm9).

We mainly add Minkowski norm and inner product in [`collate.py`](./collate.py), and use distributed data loaders and samplers  in [`dataset.py`](./dataset.py) to accommodate distributed data-parallel training.

### References
[1] https://github.com/fizisist/LorentzGroupNetwork/tree/master/src/lgn/data

[2] https://github.com/vgsatorras/egnn/tree/main/qm9