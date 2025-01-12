#!/usr/bin/env python3
import argparse
import os
import json
import csv
import glob
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from contextlib import suppress
from torch.utils.data.dataloader import DataLoader
from timm.models import create_model, load_checkpoint, is_model, list_models
from timm.data import create_dataset, resolve_data_config, RealLabelsImagenet
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging, set_jit_legacy

import models
from metrics import *
from datasets.mp_liver_dataset import MultiPhaseLiverDataset, create_loader
from torchvision.transforms.functional import to_pil_image
from matplotlib import cm
from PIL import Image
import cv2

from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve,auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

has_apex = False
try:
    from apex import amp
    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('validate')


parser = argparse.ArgumentParser(description='LLD-MMRI2023 Validation')

parser.add_argument('--img_size', default=(16, 128, 128), type=int, nargs='+', help='input image size.')
parser.add_argument('--crop_size', default=(14, 112, 112), type=int, nargs='+', help='cropped image size.')
parser.add_argument('--data_dir', default='./images/', type=str)
parser.add_argument('--val_anno_file', default='./labels/test.txt', type=str)
parser.add_argument('--val_transform_list', default=['center_crop'], nargs='+', type=str)
parser.add_argument('--model', '-m', metavar='NAME', type=str, nargs='+', default='resnet50',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--num-classes', type=int, default=7,
                    help='Number classes in dataset')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, nargs='+', metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--test-pool', dest='test_pool', action='store_true',
                    help='enable test time pool')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--tf-preprocessing', action='store_true', default=False,
                    help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--legacy-jit', dest='legacy_jit', action='store_true',
                    help='use legacy jit mode for pytorch 1.5/1.5.1/1.6 to get back fusion performance')
parser.add_argument('--results-dir', default='', type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')
parser.add_argument('--score-dir', default='', type=str, metavar='FILENAME',
                    help='Output csv file for validation score (summary)')
parser.add_argument('--mode', type=str, default='trilinear', help='interpolate mode (trilinear, tricubic)')
parser.add_argument('--return_visualization', action='store_true', default=False, help='if return_visualization')
parser.add_argument('--return_hidden', action='store_true', default=False, help='if return_model_hidden')
parser.add_argument('--handcraft_feature',action='store_true', default=False, help='if add handcraft_feature')
parser.add_argument('--patch_size',default=(2, 2, 2), type=int, nargs='+', help='the first conv kernal size of uniformer.')

def validate(args):
    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint
    amp_autocast = suppress  # do nothing
    if args.amp:
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
        else:
            _logger.warning("Neither APEX or Native Torch AMP is available.")
    assert not args.apex_amp or not args.native_amp, "Only one AMP mode should be set."
    if args.native_amp:
        amp_autocast = torch.cuda.amp.autocast
        _logger.info('Validating in mixed precision with native PyTorch AMP.')
    elif args.apex_amp:
        _logger.info('Validating in mixed precision with NVIDIA APEX AMP.')
    else:
        _logger.info('Validating in float32. AMP not enabled.')

    if args.legacy_jit:
        set_jit_legacy()

    # create model
    if isinstance(args.model,list):
        assert isinstance(args.checkpoint,list),'if multimodel, must have corresponding checkpoints'
        assert len(args.model) == len(args.checkpoint)
        model_list = []
        for model_name,checkpoint_path in zip(args.model,args.checkpoint):
            # create model
            model = create_model(
                model_name,
                pretrained=args.pretrained,
                num_classes=args.num_classes,
                img_size = args.crop_size[-1],
                patch_size = args.patch_size,
                pretrained_cfg=None,
                handcraft_branch = args.handcraft_feature,
                return_hidden = args.return_hidden,
                return_visualization = args.return_visualization,
                )
            if args.num_classes is None:
                assert hasattr(
                    model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
                args.num_classes = model.num_classes
            
            load_checkpoint(model, checkpoint_path, args.use_ema)

            param_count = sum([m.numel() for m in model.parameters()])
            _logger.info('Model %s created, param count: %d' %
                        (model_name, param_count))
            
            model = model.cuda()

            if args.apex_amp:
                model = amp.initialize(model, opt_level='O1')

            if args.num_gpu > 1:
                model = torch.nn.DataParallel(
                    model, device_ids=list(range(args.num_gpu)))
            
            model.eval()

            model_list.append(model)
    else:
        # create model
        model = create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            img_size = args.crop_size[-1],
            patch_size = args.patch_size,
            pretrained_cfg=None,
            handcraft_branch = args.handcraft_feature,
            return_hidden = args.return_hidden,
            return_visualization = args.return_visualization,
                            )
        if args.num_classes is None:
            assert hasattr(
                model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
            args.num_classes = model.num_classes
        if args.checkpoint:
            load_checkpoint(model, args.checkpoint, args.use_ema)

        param_count = sum([m.numel() for m in model.parameters()])
        _logger.info('Model %s created, param count: %d' %
                    (args.model, param_count))

        model = model.cuda()
        if args.apex_amp:
            model = amp.initialize(model, opt_level='O1')

        if args.num_gpu > 1:
            model = torch.nn.DataParallel(
                model, device_ids=list(range(args.num_gpu)))
            
        model.eval()

    criterion = nn.CrossEntropyLoss().cuda()

    dataset = MultiPhaseLiverDataset(args, is_training=False)

    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        num_workers=args.workers,
                        pin_memory=args.pin_mem,
                        shuffle=False)

    batch_time = AverageMeter()

    predictions_total = []
    labels_total = []

   
    end = time.time()
    for idx,model in enumerate(model_list):
        model_name = args.model[idx]
        predictions,labels = val_oridataset(model,model_name,dataset,loader,args)
        predictions_total.append(predictions)
        labels_total.append(labels)
        batch_time.update(time.time() - end)
        end = time.time()

    predictions_total_ = [torch.cat(i) for i in predictions_total]
    predictions_total_ = torch.stack(predictions_total_).mean(dim=0)
    predictions_total_ = [row.view(1, -1) for row in predictions_total_]

    font_size = 20

    if args.return_hidden:
        labels_total_ = torch.cat(labels_total[0][:-1]).reshape(-1)
        labels_total_ = torch.cat([labels_total_,labels_total[0][-1]])
        print("Computing t-SNE embedding")
        np_embedding_list = [item.cpu().detach().numpy() for item in predictions_total_]
        X_embedding = np.concatenate(np_embedding_list, axis=0)
        tsne2d = TSNE(n_components=2, init='pca', random_state=0)

        X_tsne_2d = tsne2d.fit_transform(X_embedding)
        plot_embedding_2d(X_tsne_2d[:,0:2],labels_total_.cpu().detach().numpy(),"t-SNE 2D",args.results_dir,fontsize=font_size)

    ### plot auc curve ###
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    labels_total_ = torch.cat(labels_total[0][:-1]).reshape(-1)
    labels_total_ = torch.cat([labels_total_,labels_total[0][-1]])
    labels_total_ = labels_total_.cpu().detach().numpy()

    predictions_total__ = torch.cat(predictions_total_)
    predictions_total__ = predictions_total__.cpu().detach().numpy()

    for i in range(args.num_classes):
        index = np.where(labels_total_==i)[0]
        fpr[i],tpr[i],thresholds = roc_curve(labels_total_,predictions_total__[:,i],pos_label=i)
        roc_auc[i] = auc(fpr[i],tpr[i])
    
    plt.figure(figsize=(10,7))
    colors = ['red','blue','green','cyan','yellow','magenta','pink','purple','gold','orange']
    for i,color in zip(range(args.num_classes),colors):
        plt.plot(fpr[i],tpr[i],color=color,lw=2,label='ROC curve of class {0} (area = {1:0.3f})'.format(i,roc_auc[i]))
    plt.plot([0,1],[0,1], color='navy',linestyle='--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate',fontsize=font_size)
    plt.ylabel('True Positive Rate',fontsize=font_size)
    plt.title('ROC Curve for total classes',fontsize=font_size)
    plt.legend(loc="lower right",fontsize=font_size-4)
    plt.savefig(os.path.join(args.results_dir,'roc.jpg'))

    ### plot micro-auc ###
    label_binary = label_binarize(labels_total_,classes=np.arange(args.num_classes))
    fpr_micro,tpr_micro,thresholds_micro = roc_curve(label_binary.ravel(),predictions_total__.ravel())
    roc_auc_micro = auc(fpr_micro,tpr_micro)

    plt.figure(figsize=(10,7))
    plt.plot(fpr_micro,tpr_micro,color='deeppink',lw=2,label='micro-average ROC curve (area = {0:0.3f})'.format(roc_auc_micro))
    plt.plot([0,1],[0,1], color='navy',linestyle='--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate',fontsize=font_size)
    plt.ylabel('True Positive Rate',fontsize=font_size)
    plt.title('micro-average ROC curve',fontsize=font_size)
    plt.legend(loc="lower right",fontsize=font_size-4)
    plt.savefig(os.path.join(args.results_dir,'micro_roc.jpg'))

    evaluation_metrics, evaluation_scores, confusion_matrix = compute_metrics(predictions_total_, labels_total[0], criterion, args)
    # plot_confusion_matrix(confusion_matrix,['Hepatic_hemangioma', 'Intrahepatic_cholangiocarcinoma', 'Hepatic_abscess', 'Hepatic_metastasis', 
    #                   'Hepatic_cyst', 'Focal_nodular_hyperplasia', 'Hepatocellular_carcinoma'],os.path.join(args.results_dir,'confusion_matrix.png'),fontsize=font_size)
    plot_confusion_matrix(confusion_matrix,['a', 'b', 'c', 'd', 
                      'e', 'f', 'g'],os.path.join(args.results_dir,'confusion_matrix.png'),fontsize=font_size)

    return evaluation_metrics, evaluation_scores

def plot_embedding_2d(X, y, title=None, save_path=None, fontsize=18):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set3(y[i] / 7.),
                 fontdict={'weight': 'bold', 'size': fontsize})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title,fontsize=fontsize)
    # plt.show()
    plt.savefig(os.path.join(save_path,'tsne.png'))

def val_oridataset(model, model_name, dataset, loader, args):
    amp_autocast = suppress

    predictions = []
    labels = []

    pbar = tqdm(total=len(dataset))
    with torch.no_grad():
        for batch_idx, item in enumerate(loader):
            input = item['images']
            target = item['labels']
            benign_malignant_target = item['benign_malignant_labels']
            handcraft_input = None
            input, target = input.cuda(), target.cuda()
            benign_malignant_target = benign_malignant_target.cuda()
            if args.handcraft_feature:
                handcraft_input = item['handcraft_feature']
                handcraft_input = handcraft_input.cuda()
            
            # compute output
            with amp_autocast():
                if model_name == "uniformer_small_original" or model_name == "uniformer_base_original" \
                or model_name == "uniformer_xs_original" or model_name == "uniformer_xxs_original" or model_name == 'two_branch_uniformer' or model_name == "three_branch_uniformer" \
                or model_name == "multiphase_Bilstm_model" or model_name == "eight_branch_uniformer" or model_name == "multiphase_Bilstm_model_two_branch" \
                or model_name == "multiphase_Bilstm_model_four_branch" or model_name == "latent_fusion_uniformer_base" or model_name == "multiphase_Bilstm_model_one_branch" \
                or model_name == "lstm_uniformer_base" or model_name == "latent_fusion_lstm_uniformer_base" or model_name == "latent_fusion_cls_uniformer_v2_base":
                    output,visualizations = model(input,handcraft_input)
                elif model_name == "multiphase_Bilstm_model_2cls_aug":
                    output,visualizations,aug_cls = model(input,handcraft_input)
                elif model_name == "resnet3d_50_classify" or model_name == "swin_transformer_3d_base" or model_name == "inception_resnet_v2_classify":
                    output = model(input)
                else:
                    raise ValueError('invalid model input')

            if args.return_visualization:
                ## try vis self_attn_map ###

                for img_idx in range(input.shape[0]):
                    for modal in range(input.shape[1]):
                        cur_img = input[img_idx][modal] # Z,H,W
                        for branch_idx, item in enumerate(visualizations):
                            cur_attn_map = np.mean(item[0],axis=0) # Z,H,W
                            cur_attn_map_tensor = torch.from_numpy(cur_attn_map).unsqueeze(0).unsqueeze(0)
                            cur_attn_map_tensor_resize = F.interpolate(cur_attn_map_tensor, size=input.shape[-3:], mode='trilinear', align_corners=True).squeeze(0).squeeze(0)
                            for z in range(input.shape[2]):
                                cur_img_z = cur_img[z]
                                cur_img_z = cur_img_z.cpu().numpy()
                                cur_img_z = np.repeat(cur_img_z[:, :, np.newaxis], 3, axis=-1)
                                cur_img_z = (cur_img_z-cur_img_z.min())/(cur_img_z.max()-cur_img_z.min())
                                cur_img_z = (cur_img_z * 255).astype(np.uint8)
                                mask = to_pil_image(cur_attn_map_tensor_resize[z], mode='F')
                                cmap = cm.get_cmap('jet')
                                overlay = (255 * cmap(np.asarray(mask) ** 2)[:, :, :3]).astype(np.uint8).transpose(1,0,2)[:,:,::-1]

                                out_img = cur_img_z*0.5+overlay*0.5
                                concat_img = np.concatenate([cur_img_z,overlay,out_img],axis=1)
                                cv2.imwrite('/home/mdisk3/bianzhewu/medical_repertory/miccai2023_extension/main/vis/modal_%s_branch_%s_slice_%s.jpg'%(modal,branch_idx,z),concat_img)

            predictions.append(output)
            labels.append(target)
            pbar.update(args.batch_size)
        pbar.close()

    return predictions, labels

def compute_metrics(outputs, targets, loss_fn, args):
    
    outputs = torch.cat(outputs, dim=0).detach()
    targets = torch.cat(targets, dim=0).detach()
    pred_score = torch.softmax(outputs, dim=1)
    loss = loss_fn(outputs, targets).cpu().item()
    outputs = outputs.cpu().numpy()
    targets = targets.cpu().numpy()
    pred_score = pred_score.cpu().numpy()
    acc = ACC(outputs, targets)
    f1 = F1_score(outputs, targets)
    recall = Recall(outputs, targets)
    # specificity = Specificity(outputs, targets)
    precision = Precision(outputs, targets)
    kappa = Cohen_Kappa(outputs, targets)
    report = cls_report(outputs, targets)
    cm = confusion_matrix(outputs, targets)
    metrics = OrderedDict([
        ('acc', acc),
        ('f1', f1),
        ('recall', recall),
        ('precision', precision),
        ('kappa', kappa),
        ('confusion matrix', cm),
        ('classification report', report),
    ])
    return metrics, pred_score, cm

def write_results2txt(results_dir, results):
    results_file = os.path.join(results_dir, 'results.txt')
    file = open(results_file, 'w')
    file.write(results)
    file.close()

def write_score2json(score_info, args):
    score_info = score_info.astype(float)
    score_list = []
    anno_info = np.loadtxt(args.val_anno_file, dtype=np.str_)
    for idx, item in enumerate(anno_info):
        id = item[0].rsplit('/', 1)[-1]
        label = int(item[1])
        score = list(score_info[idx])
        pred = score.index(max(score))
        pred_info = {
            'image_id': id,
            'label': label,
            'prediction': pred,
            'score': score,
        }
        score_list.append(pred_info)
    json_data = json.dumps(score_list, indent=4)
    file = open(os.path.join(args.results_dir, 'score.json'), 'w')
    file.write(json_data)
    file.close()

def plot_confusion_matrix(confusion_matrix, classes_name, output_path, fontsize = 18):
    num_classes = len(classes_name)
    plt.figure(figsize=(12,10))
    plt.imshow(confusion_matrix,interpolation='nearest',cmap=plt.cm.Blues)
    plt.colorbar(shrink=1.0)
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks,classes_name,rotation=0,ha='center',fontsize=fontsize+8) # rotation参数可以旋转xticks,ha参数可以改变xticks的位置
    plt.yticks(tick_marks,classes_name,fontsize=fontsize+8)

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j,i,str(confusion_matrix[i,j]),
                    horizontalalignment='center',color='white' if confusion_matrix[i,j]>np.max(confusion_matrix)/2 else 'black', fontsize=fontsize)

    plt.title('Confusion Matrix',fontsize=fontsize+8)
    plt.xlabel('Predicted Label',fontsize=fontsize+8)
    plt.ylabel('True Label',fontsize=fontsize+8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)

def main():
    setup_default_logging()
    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    results, score = validate(args)
    output_str = 'Test Results:\n'
    for key, value in results.items():
        if key == 'confusion matrix':
            output_str += f'{key}:\n {value}\n'
        elif key == 'classification report':
            output_str += f'{key}:\n {value}\n'
        else:
            output_str += f'{key}: {value}\n'
    write_results2txt(args.results_dir, output_str)
    write_score2json(score, args)
    print(output_str)

if __name__ == '__main__':
    main()