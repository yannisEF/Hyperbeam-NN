# Hyperbeam NN

An interactive hyper-beam based visualization method of n-th dimensional reward landscapes, fully compatible with [Stable-baselines 3](https://stable-baselines3.readthedocs.io/en/master/)'s policies.

<p align="middle">
  <img src="https://github.com/yannisEF/NNMRI/assets/49323355/4f972e98-1305-4012-89c3-8dffedbe390a" width="200" />
  <img src="https://github.com/yannisEF/NNMRI/assets/49323355/4fdc8c50-778e-4f0c-9404-47002ce26722" width="200" /> 
  <img src="https://github.com/yannisEF/NNMRI/assets/49323355/dddbba50-51f5-499c-9a6d-de9584de04d2" width="200" /> 
</p>

Manipulate a 3D example of the tool in the [utils_sample.py](./utils_sample.py) file, see the [project report](./BCS541_ELRHARBI-FLEURY_YANNIS.pdf) for more details.

```bash
python3 utils_sample.py
```

<p align="middle">
  <img src="https://github.com/yannisEF/NNMRI/assets/49323355/96c32e9c-18f5-4299-b1b7-eeec0c30a8a5" width="300" />
  <img src="https://github.com/yannisEF/NNMRI/assets/49323355/108fcdee-c84b-413f-9baf-03ea07757a01" width="300" /> 
</p>

## How to use

User parameters available in parameters.json

Train a model using [model_train.py](./model_train.py), results will be saved in the folder [Models](./Models) within the appropriate subfolder.

```bash
python3 model_train.py --parameters parameters.json
```

Prepare the hyper-beam using [model_prepare.py](./model_prepare.py), results will be saved in the folder [Results](./Results) as a .pkl file of the appropriate name.

```bash
python3 model_prepare.py --parameters parameters.json
```
:warning: There appears to be an issue with NVIDIA drivers that prevents the use of multi-processing. For an unknown reason, this resolves on its own after a while (debugging needed...). If this happens, try to run a few SB3 policies in your python environment.

Visualize the results using [main.py](./main.py) and enjoy.
```bash
python3 main.py --to_show wide_test
```
