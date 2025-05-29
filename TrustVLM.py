import argparse

import time

from copy import deepcopy

from PIL import Image
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F
from scipy.stats import entropy
import torchvision.datasets as tv_datasets

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

from clip.custom_clip import get_model
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
from sklearn import metrics
from transformers import AutoModel

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

to_np = lambda x: x.data.cpu().numpy()

def calc_aurc_eaurc(softmax_max, correct):
    correctness = np.array(correct)

    sort_values = sorted(zip(softmax_max[:], correctness[:]), key=lambda x: x[0], reverse=True)
    sort_softmax_max, sort_correctness = zip(*sort_values)
    risk_li, coverage_li = coverage_risk(sort_softmax_max, sort_correctness)
    aurc, eaurc = aurc_eaurc(risk_li)

    return aurc, eaurc

def calc_fpr_aupr(softmax_max, correct):
    correctness = np.array(correct)

    fpr, tpr, thresholds = metrics.roc_curve(correctness, softmax_max)
    auroc = metrics.auc(fpr, tpr)
    idx_tpr_95 = np.argmin(np.abs(tpr - 0.95))
    fpr_in_tpr_95 = fpr[idx_tpr_95]
    tnr_in_tpr_95 = 1 - fpr[np.argmax(tpr >= .95)]

    precision, recall, thresholds = metrics.precision_recall_curve(correctness, softmax_max)
    aupr_success = metrics.auc(recall, precision)
    aupr_err = metrics.average_precision_score(-1 * correctness + 1, -1 * softmax_max)

    return auroc, aupr_success, aupr_err, fpr_in_tpr_95, tnr_in_tpr_95

def coverage_risk(confidence, correctness):
    risk_list = []
    coverage_list = []
    risk = 0
    for i in range(len(confidence)):
        coverage = (i + 1) / len(confidence)
        coverage_list.append(coverage)

        if correctness[i] == 0:
            risk += 1

        risk_list.append(risk / (i + 1))

    return risk_list, coverage_list

def aurc_eaurc(risk_list):
    r = risk_list[-1]
    risk_coverage_curve_area = 0
    optimal_risk_area = r + (1 - r) * np.log(1 - r)
    for risk_value in risk_list:
        risk_coverage_curve_area += risk_value * (1 / len(risk_list))

    aurc = risk_coverage_curve_area
    eaurc = risk_coverage_curve_area - optimal_risk_area
    return aurc, eaurc


def main():
    args = parser.parse_args()
    set_random_seed(args.seed)

    assert args.gpu is not None
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu
    set_random_seed(args.seed)
    print("Use GPU: {} for training".format(args.gpu))

    if args.test_sets in fewshot_datasets:
        classnames = eval("{}_classes".format(args.test_sets.lower()))
    elif args.test_sets == 'cifar10':
        classnames = tv_datasets.CIFAR10(root="./data", train=False, download=True).classes
    elif args.test_sets == 'cifar100':
        classnames = tv_datasets.CIFAR100(root="./data", train=False, download=True).classes
    else:
        classnames = imagenet_classes

    model = get_model(args.arch, args.test_sets, args.gpu, args.n_ctx, args.ctx_init)
    
    print("=> Model created: visual backbone {}".format(args.arch))
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    model_name = "facebook/dinov2-base" 
    dino_model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino_model = dino_model.to(device).eval()

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    datasets = args.test_sets.split("/")
    results = {}
    for set_id in datasets:
        data_transform = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            normalize,
        ])
        batchsize = args.batch_size

        print("evaluating: {}".format(set_id))

        model.reset_classnames(classnames, args.arch)
        num_classes = len(classnames)

        val_dataset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode)
        print("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batchsize, shuffle=True,
                    num_workers=args.workers, pin_memory=True)

        retrieval_dataset = build_dataset(set_id, data_transform, args.data, mode='train', n_shot=args.n_shot)

        results[set_id] = misd_eval(val_loader, model, args, retrieval_dataset, dino_model, num_classes)
        del val_dataset, val_loader
        try:
            print("=> Acc. on testset [{}]: @1 {}/ @5 {}".format(set_id, results[set_id][0], results[set_id][1]))
        except:
            print("=> Acc. on testset [{}]: {}".format(set_id, results[set_id]))

    print("======== Result Summary ========")
    print("\t\t [set_id] \t\t Top-1 acc. \t\t Top-5 acc.")
    for id in results.keys():
        print("{}".format(id), end="	")
    print("\n")
    for id in results.keys():
        print("{:.2f}".format(results[id][0]), end="	")
    print("\n")


def misd_eval(val_loader, model, args, retrieval_dataset, dino_model, num_classes):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

    list_score = []
    list_correct = []

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    class_indices = [[] for _ in range(num_classes)]
    for idx, (_, label) in enumerate(retrieval_dataset):
        class_indices[label].append(idx)

    # create visual prototypes
    list_features = []
    for class_idx in range(num_classes):
        indices = class_indices[class_idx]
        images_with_label_i = [retrieval_dataset[idx][0] for idx in indices]
        images_i = torch.stack(images_with_label_i)
        images_i = images_i.cuda(args.gpu, non_blocking=True)

        with torch.no_grad():
            image_features_i = dino_model(images_i).last_hidden_state.mean(dim=1)
            image_features_i = torch.mean(image_features_i, dim=0, keepdim=True)
            image_features_i = image_features_i / image_features_i.norm(dim=-1, keepdim=True)
            list_features.append(image_features_i)

    list_features = torch.cat(list_features, dim=0)

    model.eval()
    for i, (images, target) in enumerate(val_loader):
        assert args.gpu is not None
        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
            image = images[0]
        else:
            if len(images.size()) > 4:
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(args.gpu, non_blocking=True)
            image = images
        target = target.cuda(args.gpu, non_blocking=True)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = model(image)
                image_features = dino_model(image).last_hidden_state.mean(dim=1)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                if args.use_trustvlm_v2:
                    output = F.softmax(output, dim=1)
                    output_v = image_features @ list_features.t()
                    output_v = F.softmax(output_v/args.T, dim=1)
                    output = output + args.dino_ratio * output_v

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
                
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        
        if args.use_trustvlm_v2:
            smax = to_np(output)
        else:
            smax = to_np(F.softmax(output/args.T, dim=1))

        # image-text score
        MSP_score = np.max(smax, axis=1) 

        pred = output.data.max(1, keepdim=True)[1]
        for j in range(len(pred)):
            if pred[j] == target[j]:
                cor = 1
            else:
                cor = 0
            list_correct.append(cor)

            # image-image score
            similarity = image_features[j] @ list_features[pred[j]].T
            list_score.append(MSP_score[j] + args.dino_ratio * to_np(torch.max(similarity))) 

        if (i+1) % args.print_freq == 0:
            progress.display(i)

    list_score = np.array(list_score)

    aurc, eaurc = calc_aurc_eaurc(list_score, list_correct)
    auroc, aupr_success, aupr, fpr, tnr = calc_fpr_aupr(list_score, list_correct)

    print("AURC {0:.2f}".format(aurc * 1000))
    print("AUROC {0:.2f}".format(auroc * 100))
    print('FPR95 {0:.2f}'.format(fpr * 100))

    progress.display_summary()

    return [top1.avg, top5.avg]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TrustVLM')
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='A/R/V/K/I', help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-p', '--print-freq', default=200, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default=None, type=str, help='init tunable prompts')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--T', type=float, default=1.0, help='temperature parameter')
    parser.add_argument('--dino_ratio', type=float, default=1.0, help='dino_ratio')
    parser.add_argument('--use_trustvlm_v2', action='store_true')
    parser.add_argument('--n_shot', default=16, type=int, help='number of shots')

    main()