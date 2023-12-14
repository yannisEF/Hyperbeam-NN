# NNMRI

An interactive hyper-beam based visualization method of n-th dimensional reward landscapes. See the BCS541_ELRHARBI-FLEURY_YANNIS.pdf for more details.

Fully compatible with [Stable-baselines 3](https://stable-baselines3.readthedocs.io/en/master/)'s policies.

Manipulate a 3D example of the tool in the utils_sample.py file.

```bash
python3 utils_sample.py
```

## How to use

User parameters available in parameters.json

Train a model using model_train.py, results will be saved in the folder "Models/" within the appropriate subfolder.

```bash
python3 model_train.py --parameters parameters.json
```

Prepare the hyper-beam using model_prepare.py, results will be saved in the folder "Results/" as a .pkl file of the appropriate name.

```bash
python3 model_prepare.py --parameters parameters.json
```

Warning : there appears to be an issue with NVIDIA drivers that prevents the use of multi-processing. For an unknown reason, this resolves on its own after a while (debugging needed...). If this happens, try to run a few SB3 policies in your python environment.

Visualize the results using main.py and enjoy.
```bash
python3 main.py --to_show wide_test
```