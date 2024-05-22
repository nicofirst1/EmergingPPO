# EmergingPPO

## Installation

The project can be installed with the installation script from the working dir
```bash
bash scripts/install.sh
```

### Pip installation
It can also be installed with pip in edit mode with 

```bash
pip install -e .
```


If you want to install the dev  :
```bash 
pip install -e .[dev]
```

## Testing
In order to test you need to install the dev packages as shown above. 
then use
```bash
pytest
```

## useful links
- PPO code : https://github.com/huggingface/trl/blob/v0.7.9/trl/trainer/ppo_trainer.py
- PPO example:  https://huggingface.co/docs/trl/ppo_trainer
- Base paper:  https://openreview.net/pdf?id=-Yzz6vlX7V-
- Paper repo:  https://github.com/hcoxec/variable_compositionality