# blip2_finetune

# multimodal-cot
Repo for summer23 internship project on multimodal CoT

## Installation


```
git clone --recurse-submodules https://github.com/tejas1995/blip2_finetune.git
cd src/
```

Create conda environment
```
conda create -n mcot python=3.9
conda activate mcot
```

Install packages:
```
pip install -e modules/LAVIS/
python -m spacy download en_core_web_sm
pip install openai wandb nltk accelerate
```

Install NLTK resources:
```
import nltk; nltk.download('popular')
```

## Training

Modify paths of `output_dir` and `wandb_config_file` in L305-306 of `train.py` and, from `src/`, run the following command:
```
accelerate -W ignore launch train.py \
    --model_config_file configs/model_configs/blip2_flant5xl.yaml \
    --training_config_file configs/training_configs/aokvqa_train_blip2.yaml \
    --mode q2a
``` 

You will have to first configure `accelerate` to use 2 GPUs on a single machine. It's been a while since I did this, but the documentation should tell you how.