
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.imagnet_prompts import imagenet_classes
from data.fewshot_datasets import fewshot_datasets
from data.cls_to_names import *
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask

from transformers import AutoModel, AutoTokenizer


class ClipTestTimeTuning(nn.Module):
    def __init__(self, device, classnames):
        super(ClipTestTimeTuning, self).__init__()
        self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224").eval().to(device)
        self.logit_scale = 100
        self.tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")

        candidate_labels = [name.replace("_", " ") for name in classnames]
        texts = [f'This is a photo of {label}.' for label in candidate_labels]
        if len(classnames) > 200:
            chunk_size = 200
            self.text_features = []

            for i in range(0, len(classnames), chunk_size):
                # Tokenize and move to GPU
                text_inputs = self.tokenizer(
                    texts[i:i + chunk_size], 
                    padding="max_length", 
                    return_tensors="pt"
                ).to(device)
                
                # Get text features
                with torch.no_grad():
                    text_features = self.model.get_text_features(**text_inputs)
                
                # Normalize and store
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                self.text_features.append(text_features)
                
                # Clear GPU cache
                del text_inputs, text_features
                torch.cuda.empty_cache()

            # Concatenate all features
            self.text_features = torch.cat(self.text_features, dim=0)
        else:
            text_inputs = self.tokenizer(texts, padding="max_length", return_tensors="pt").to(device)
            self.text_features = self.model.get_text_features(**text_inputs)
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        

    def inference(self, image):
        with torch.no_grad():
            image_features = self.model.get_image_features(image)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        logits = self.logit_scale * image_features @ self.text_features.t()

        return logits

    def forward(self, input):
        return self.inference(input)


def get_coop(test_set, device, learned_cls=False):
    if test_set in fewshot_datasets:
        classnames = eval("{}_classes".format(test_set.lower()))
    elif test_set == 'bongard':
        if learned_cls:
            classnames = ['X', 'X']
        else:
            classnames = ['True', 'False']
    else:
        #classnames = imagenet_classes
        classnames_all = imagenet_classes
        classnames = []
        if test_set in ['A', 'R', 'V']:
            label_mask = eval("imagenet_{}_mask".format(test_set.lower()))
            if test_set == 'R':
                for i, m in enumerate(label_mask):
                    if m:
                        classnames.append(classnames_all[i])
            else:
                classnames = [classnames_all[i] for i in label_mask]
        else:
            classnames = classnames_all

    model = ClipTestTimeTuning(device, classnames)

    return model

