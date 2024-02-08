import torch
from torch import nn
from lavis.models import load_model_and_preprocess, load_preprocess
from lavis.common.registry import registry
from omegaconf import OmegaConf

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

def load_model_with_custom_configs(name, config, is_eval=False, device="cpu"):
    model_cls = registry.get_model_class(name)
    cfg = OmegaConf.create(config)
    if 'model_config' not in config:
        model, _, _ = load_model_and_preprocess(
            name=config['name'],
            model_type=config['type'],
            is_eval=is_eval,
            device=device,
        )
    else:
        model = model_cls.from_config(cfg.model_config)

    preprocess_cfg = cfg.preprocess_config
    vis_processors, txt_processors = load_preprocess(preprocess_cfg)

    if is_eval:
        model.eval()

    return model.to(device), vis_processors, txt_processors


class BLIP2(nn.Module):
    def __init__(self, config, device):
        super(BLIP2, self).__init__()

        load_config = config["pt_model_load"]
        self.model, self.vis_processors, self.text_processors = load_model_with_custom_configs(
                name=load_config['name'],
                config=load_config,
                device=device
            )
        
        #preproc_config = OmegaConf.create(config['preprocess_config'])
        #self.vis_processors, self.text_processors = load_preprocess(preproc_config)
        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad is True)

        self.device = device
        logger.info(f"Loaded BLIP-2 (model={config['model_shorthand']})!")
        logger.info("Model size: {:.2f}B parameters ({:.2f}B trainable)".format(num_params*10**-9, num_trainable_params*10**-9))

    def preprocess_batch(self, batch, mode='train'):
        images = batch['images']
        text_inputs = batch['text_inputs']
        text_outputs = batch['text_outputs']
        
        processed_images = torch.stack([self.vis_processors[mode](img) for img in images], dim=0).to(self.device)
        processed_text_inputs = [self.text_processors[mode](txt) for txt in text_inputs]
        processed_text_outputs = [self.text_processors[mode](txt) for txt in text_outputs]
        samples = {
            "image": processed_images,
            "text_input": processed_text_inputs,
            "text_output": processed_text_outputs,
            "prompt": text_inputs,
            "score_dict": batch['score_dict'],
            "qids": batch['qids'],
        }
        return samples

    def set_inference_params(self, inference_params):
        self.inference_params = inference_params

    def forward(self, batch):
        output_dict = self.model(batch)
        loss = output_dict["loss"]
        return loss

    def generate(self, batch):
        with torch.no_grad():
            output = self.model.generate(batch, **self.inference_params)
        output = [x.lower() for x in output]
        return output

    def get_vqa_pred(self, question, raw_image):
        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        prompt = f"Question: {question} Short answer: "
        answer = self.model.generate({"image": image, "prompt": prompt}, length_penalty=-1.0)[0].lower()
        return answer

    def ask(self, raw_image, question, length_penalty=-1.0):
        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        prompt = question
        answer = self.model.generate({"image": image, "prompt": prompt}, length_penalty=length_penalty)[0].lower()
        return answer

    def caption(self, raw_image):
        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        prompt = f"a photo of "
        answer = self.model.generate({"image": image, "prompt": prompt}, length_penalty=-1.0)[0].lower()
        return answer