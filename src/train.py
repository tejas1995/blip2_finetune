import os
import json
import pdb
import yaml
from tqdm import tqdm
import argparse
import random
import spacy
import logging
import copy
import datetime

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from transformers import get_polynomial_decay_schedule_with_warmup
from accelerate import Accelerator

from data.aokvqa import AOKVQADataset
from models import MODEL_REGISTRY

from utils.okvqa_utils import postprocess_ok_vqa_generation, lemmatize
from utils.wandb import wandb_logger

from nltk.translate.bleu_score import sentence_bleu

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

def extract_answer(output):
    output = output.split('answer')[-1].strip()
    if output.startswith(':'):
        output = output.split(':')[1].strip()
    return output

torch.distributed.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=5400))

class Trainer:

    def __init__(self, args, training_config, model_config):
        self.args = args
        self.num_epochs = training_config['num_epochs']
        self.batch_size = training_config['batch_size']
        self.inference_params = training_config['inference_params'][args.mode]

        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.is_mainproc = self.accelerator.is_local_main_process
        if self.is_mainproc:
            logger.info("Using accelerate!")
        
        self.model, vis_processors, text_processors = self.create_model(model_config)
        if args.model_init is not None:
            logger.info(f"Initializing model from checkpoint {args.model_init}")
            self.model.load_state_dict(torch.load(args.model_init))
        self.model.to(self.device)

        self.train_dataloader, self.eval_dataloader = self.create_dataloader(args, vis_processors, text_processors)
        self.loss_criterion = nn.CrossEntropyLoss(reduction='mean')
        self.optimizer, self.scheduler = self.create_optimizer(self.model, training_config['optimizer'])


        self.model, self.optimizer, self.train_dataloader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.scheduler
        )

        experiment_name = f"aokvqa_finetune_{args.mode}-{model_config['model_shorthand']}"
        if args.model_init is not None:
            experiment_name += '-q2a_init'
        if 'r' in args.mode:
            experiment_name += f"-{args.rationales_type}_rationales"
        if args.train_qids_list is not None:
            experiment_name += f"-{len(self.train_dataloader.dataset)}_train_examples"
        if args.val_qids_list is not None:
            experiment_name += f"-{len(self.eval_dataloader.dataset)}_eval_examples"

        self.output_dir = os.path.join(args.output_dir, experiment_name)
        if self.is_mainproc:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            wandb_logger.initialize(args.wandb_config_file, experiment_name)
            logger.info(f"Saving best model checkpoints to {self.output_dir}")


    def create_dataloader(self, args, vis_processors, text_processors):
        # Create train and eval dataloaders
        train_dataset = AOKVQADataset(split='train', mode=args.mode, vis_processors=vis_processors, text_processors=text_processors)
        eval_dataset = AOKVQADataset(split='val', mode=args.mode, vis_processors=vis_processors, text_processors=text_processors)

        # Filter train dataset by qids and Set rationales for training dataset, if needed
        if args.train_qids_list is not None:
            with open(args.train_qids_list) as f:
                train_qids_list = f.read().splitlines()
                train_qids_list = [int(qid) for qid in train_qids_list]
                train_dataset.filter_by_qids(train_qids_list)
        if args.train_rationales_source is not None:
            train_rationales_data = json.load(open(args.train_rationales_source))['instances']
            assert len(train_rationales_data) == len(train_dataset)
            qid2rationale = {d['qid']: d['rationale'] for d in train_rationales_data}
            train_dataset.set_rationales(qid2rationale)

        # Filter val dataset by qids and Set rationales for val dataset, if needed
        if args.val_qids_list is not None:
            with open(args.val_qids_list) as f:
                val_qids_list = f.read().splitlines()
                val_qids_list = [int(qid) for qid in val_qids_list]
                eval_dataset.filter_by_qids(val_qids_list)
        if args.val_rationales_source is not None:
            val_rationales_data = json.load(open(args.val_rationales_source))['instances']
            assert len(val_rationales_data) == len(eval_dataset)
            qid2rationale = {d['qid']: d['rationale'] for d in val_rationales_data}
            eval_dataset.set_rationales(qid2rationale)

        train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=train_dataset.aokvqa_collate_fn
        )
        eval_dataloader = torch.utils.data.DataLoader(
            dataset=eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=eval_dataset.aokvqa_collate_fn
        )
        return train_dataloader, eval_dataloader

    def create_model(self, model_config):
        # Create model and get vis+text processors
        model_class = MODEL_REGISTRY[model_config['model_class']]
        model = model_class(config=model_config, device=self.device)
        model.set_inference_params(self.inference_params)
        model.eval()
        vis_processors = model.vis_processors
        text_processors = model.text_processors
        return model, vis_processors, text_processors

    def eval(self, model, eval_dataloader):

        model.eval()
        eval_score = 0.0
        all_eval_instances = []
        samples_completed = 0

        t = tqdm(eval_dataloader, desc="Evaluating A-OKVQA")
        for batch in t:
            samples_completed += len(batch['qids'])
            batch['image'] = batch['image'].to(self.device)

            predicted_outputs = model.module.generate(batch)
            true_outputs = batch['text_output']

            for idx, (pred_output, true_output) in enumerate(zip(predicted_outputs, true_outputs)):
                if self.args.mode in ['q2a', 'qr2a', 'q2ra']:
                    answer = extract_answer(pred_output) if self.args.mode == 'q2ra' else pred_output
                    answer = lemmatize(answer)
                    answer = postprocess_ok_vqa_generation(answer)

                    qid = batch['qids'][idx]
                    score_dict = eval_dataloader.dataset.qid2score_dict[qid]
                    score = score_dict[answer]
                    eval_score += score

                    eval_instance = {
                        'qid': qid,
                        'input': batch['prompt'][idx],
                        'true_output': true_outputs[idx],
                        'pred_output': predicted_outputs[idx],
                        'pred_answer': answer,
                        'score_dict': score_dict,
                        'score': score,
                    }
                    all_eval_instances.append(eval_instance)

                elif self.args.mode == 'q2r':
                    eval_score += sentence_bleu([true_output.lower().strip('.').split()], pred_output.split())
                    eval_instance = {
                        'qid': qid,
                        'input': batch['prompt'][idx],
                        'true_output': true_outputs[idx],
                        'pred_output': predicted_outputs[idx],
                    }
                    all_eval_instances.append(eval_instance)


            t.set_description("Evaluating A-OKVQA (score = {:.2f}%)".format(100.0*eval_score/samples_completed))

        final_eval_score = eval_score/samples_completed*100.0
        return final_eval_score, all_eval_instances

    def create_optimizer(self, model, optim_config):
        no_decay = ['bias', 'LayerNorm.weight']
        optim_config = {k: float(v) for k, v in optim_config.items()}
        optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': optim_config['weight_decay']},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=optim_config['lr'], eps=optim_config['adam_epsilon'], betas=(0.9, 0.999))
        # Create Scheduler
        max_steps = len(self.train_dataloader) * self.num_epochs
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=1000,#int(max_steps * optim_config['warmup_ratio']),
            num_training_steps=max_steps,
            lr_end=0,
            power=1,
        )
        return optimizer, scheduler

    def save_checkpoint(self, best_model):
        if self.is_mainproc:
            best_task_model_state_dict = best_model['state_dict']
            torch.save(best_task_model_state_dict, os.path.join(self.output_dir, 'model'))
            if self.is_mainproc:
                logger.info(f"Saved checkpoint to {os.path.join(self.output_dir, 'model')}!")

    def train(self):
        best_score = 0
        best_model = {
            'epoch': 0,
            'state_dict': copy.deepcopy(self.model.module.state_dict()), #model.state_dict(),
            'optimizer_state': self.optimizer.state_dict()
        }

        if self.is_mainproc:
            eval_score, _ = self.eval(self.model, self.eval_dataloader)
            logger.info("Initial eval score: {:.2f}".format(eval_score))
            wandb_logger.log({'a-okvqa': {'val_score': eval_score}})
            #pass

        self.model.zero_grad()
        for epoch in range(self.num_epochs):
            # Training loop for epoch
            #self.train_dataloader.sampler.set_epoch(epoch)

            self.accelerator.wait_for_everyone()
            self.model.train()
            for step, batch in enumerate(tqdm(self.train_dataloader, desc='Training epoch {}'.format(epoch+1))):
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    batch['image'] = batch['image'].bfloat16()
                    loss = self.model(batch).mean()
                #loss.backward()
                self.accelerator.backward(loss)

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                if (step + 1) % wandb_logger.get_log_freq() == 0:
                    log_dict = {'a-okvqa': {'loss': loss.item()}}
                    if self.is_mainproc:
                        wandb_logger.log(log_dict)
            self.accelerator.wait_for_everyone()

            # Do evaluation after epoch
            if self.is_mainproc:
                eval_score, _ = self.eval(self.model, self.eval_dataloader)
                logger.info("Evaluation after epoch {}: {:.2f}".format(epoch+1, eval_score))
                wandb_logger.log({'a-okvqa': {'val_score': eval_score}})
                if eval_score >= best_score:
                    logger.info("New best evaluation score: {:.2f}".format(eval_score))
                    best_score = eval_score
                    best_model['epoch'] = epoch
                    best_model['state_dict'] = copy.deepcopy(self.model.module.state_dict())

        if self.is_mainproc:
            logger.info("Best eval score: {:.2f}".format(best_score))
            self.save_checkpoint(best_model)
            #best_model['state_dict'] = torch.load(os.path.join(self.output_dir, 'model'))

            self.model.module.load_state_dict(best_model['state_dict'])
            eval_score, eval_instances = self.eval(self.model, self.eval_dataloader)
            logger.info("Final evaluation score: {:.2f}".format(eval_score))

            eval_file = os.path.join(self.output_dir, 'eval_instances.json')
            json.dump(eval_instances, open(eval_file, 'w'), indent=2)
            logger.info(f"Saved {len(eval_instances)} evaluation outputs to {eval_file}")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_file", type=str, required=True)
    parser.add_argument("--training_config_file", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True, choices=['q2a', 'qr2a', 'q2r', 'q2ra'])

    parser.add_argument("--model_init", type=str, default=None)

    parser.add_argument("--rationales_type", type=str, default='ground_truth')
    parser.add_argument("--train_rationales_source", type=str, default=None)
    parser.add_argument("--train_qids_list", type=str, default=None)
    parser.add_argument("--val_rationales_source", type=str, default=None)
    parser.add_argument("--val_qids_list", type=str, default=None)

    parser.add_argument("--output_dir", type=str, default='/net/nfs.cirrascale/mosaic/tejass/experiments/MCoT/finetuning/')
    parser.add_argument("--wandb_config_file", type=str, default="/net/nfs.cirrascale/mosaic/tejass/data/wandb_config.yaml")

    args = parser.parse_args()
    set_seed(42)

    if args.rationales_type != 'ground_truth':
        assert args.train_rationales_source is not None
        assert args.train_qids_list is not None


    training_config = yaml.safe_load(open(args.training_config_file))
    model_config = yaml.safe_load(open(args.model_config_file))
    trainer = Trainer(
        args=args,
        training_config=training_config,
        model_config=model_config
    )
    trainer.train()



if __name__ == '__main__':
    main()
