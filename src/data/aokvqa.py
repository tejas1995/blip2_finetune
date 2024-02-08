import torch
import os
import json
from collections import defaultdict, Counter
from torch.utils.data import Dataset
from PIL import Image
import random
import logging
from tqdm import tqdm

from utils.vqa_utils import get_score
from utils.okvqa_utils import postprocess_ok_vqa_generation, lemmatize


logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

class AOKVQADataset(Dataset):

    def __init__(self, split, mode='q2a', vis_processors=None, text_processors=None):

        data_dir = '/net/nfs.cirrascale/mosaic/tejass/data/a-okvqa'
        images_dir = '/net/nfs.cirrascale/mosaic/tejass/data/ms-coco/'
        image_filenames = os.listdir(images_dir)
        self.mode = mode
        self.split = split

        if vis_processors is not None:
            self.vis_processor = vis_processors['train'] if split == 'train' else vis_processors['eval']
        if text_processors is not None:
            self.text_processor = text_processors['train'] if split == 'train' else text_processors['eval']

        self.imageid2filename = {}
        for fn in image_filenames:
            image_id = int(fn.split('_')[-1].strip('.jpg'))
            self.imageid2filename[image_id] = os.path.join(images_dir, fn)
        self.imageids = list(set(list(self.imageid2filename.keys())))

        self.data = json.load(open(os.path.join(data_dir, f"aokvqa_v1p0_{split}.json")))
        self.qid2score_dict = []
        for i, datum in tqdm(enumerate(self.data)):
            datum['image_filename'] = self.imageid2filename[datum['image_id']]
            direct_answers = [postprocess_ok_vqa_generation(a) for a in datum['direct_answers']]
            #datum['top_answer'] = max(set(direct_answers), key=direct_answers.count)
            datum['top_answer'] = datum['choices'][datum['correct_choice_idx']]
            answer_counter = Counter(direct_answers)

            score_dict = defaultdict(int)
            for a, c in answer_counter.items():
                score_dict[a] = get_score(c)
            datum['full_score_dict'] = score_dict
            datum['qid'] = i
            self.qid2score_dict.append(score_dict)

        self.qids = list(range(len(self.data)))
        logger.info(f"Loaded A-OKVQA {split} dataset with {len(self.data)} examples!")
        if vis_processors is None or text_processors is None:
            logger.warning("Vision/text processors not set!")

    def set_rationales(self, qid2rationales):
        for i, d in enumerate(self.data):
            rationale = qid2rationales[d['qid']]
            d['rationales'] = [rationale]
        logger.info(f"Set predicted rationales for {len(self.data)} examples!")

    def filter_by_qids(self, qids_list):
        filtered_data = [d for d in self.data if d['qid'] in qids_list]
        self.data = filtered_data
        logger.info(f"Filtered A-OKVQA {self.split} dataset to {len(self.data)} examples!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data = self.data[i]
        image_filename = data['image_filename']
        raw_image = Image.open(image_filename).convert('RGB')
        
        question = data['question']
        #answer = data['choices'][ data['correct_choice_idx'] ]
        answer = data['top_answer']
        rationale = data['rationales'][0]
        if self.mode == 'q2a':
            text_input = f"Question: {question} Answer: "
            text_output = answer
        elif self.mode == 'qr2a':
            text_input = f"Question: {question} Rationale: {rationale} Answer: "
            text_output = answer
        elif self.mode == 'q2r':
            text_input = f"Question: {question} Rationale: "
            text_output = rationale
        elif self.mode == 'q2ra':
            text_input = f"Question: {question} Rationale: "
            text_output = f"{rationale} Answer: {answer}"

        score_dict = data['full_score_dict']
        choices = data['choices']
        image_id = data['image_id']
        qid = data['qid']

        return {
            'text_input': text_input,
            'text_output': text_output,
            'raw_image': raw_image,
            'score_dict': score_dict,
            'image_path': image_filename,
            'image_id': image_id,
            'question': question,
            'top_answer': answer,
            'qid': qid
        }

    def aokvqa_collate_fn(self, batch):

        images = [b['raw_image'] for b in batch]
        processed_images = torch.stack([self.vis_processor(img) for img in images], dim=0)

        text_inputs = [b['text_input'] for b in batch]
        processed_text_inputs = [self.text_processor(txt) for txt in text_inputs]

        text_outputs = [b['text_output'] for b in batch]
        processed_text_outputs = [self.text_processor(txt) for txt in text_outputs]

        score_dict = [b['score_dict'] for b in batch]
        qids = [b['qid'] for b in batch]

        collated_batch = {
            "image": processed_images,
            "text_input": processed_text_inputs,
            "text_output": processed_text_outputs,
            "prompt": text_inputs,
            "target": text_outputs,
            "qids": qids,
        }
        return collated_batch

if __name__ == '__main__':
    dataset = AOKVQADataset('train')
    import pdb; pdb.set_trace()
