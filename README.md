<div align="center">    
 
# Rule Extrapolation in Large Language Models

[//]: # ([![Paper]&#40;http://img.shields.io/badge/arxiv-cs.LG:2311.18048-B31B1B.svg&#41;]&#40;https://arxiv.org/abs/2311.18048&#41;)

[//]: # ([![Conference]&#40;http://img.shields.io/badge/CI4TS@UAI-2023.svg&#41;]&#40;https://sites.google.com/view/ci4ts2023/accepted-papers?authuser=0&#41;)

![CI testing](https://github.com/meszarosanna/rule_extrapolation/workflows/CI%20testing/badge.svg?branch=main&event=push)

</div>
 
## Description   


## How to run

### Installing dependencies

```bash
# clone the repo with submodules  
git clone --recurse-submodules https://github.com/meszarosanna/rule_extrapolation


# install the package   
cd rule_extrapolation
pip install -e .   
pip install -r requirements.txt



# install requirements for tests
pip install --requirement tests/requirements.txt --quiet

# install pre-commit hooks (only necessary for development)
pre-commit install
 ```   


### Weights and Biases sweep

```bash
# login to weights and biases
wandb login

# create sweep [spits out <sweepId>]
wandb sweep sweeps/<configFile>.yaml

# run sweep
./scripts/sweeps <sweepId>
```

## Citation   

```

@inproceedings{
 
}

```   
