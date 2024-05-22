# EmergingPPO

## Installation

The project can be installed with the installation script from the working dir
```bash
bash scripts/install.sh
```

### Dev installation

If you want to install the dev packages, use pip  :
```bash 
pip install -e .[dev]
```

Remember to install the pre-commit for automatic formatting:
```bash
pre-commit install
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