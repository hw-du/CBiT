from templates import set_template
from datasets import DATASETS
from dataloaders import DATALOADERS
from models import MODELS
from trainers import TRAINERS

import argparse


parser = argparse.ArgumentParser(description='RecPlay')

################
# Top Level
################
parser.add_argument('--mode', type=str, default='train', choices=['train'])
# train_bert
parser.add_argument('--template', type=str, default=None)

################
# Test
################
parser.add_argument('--test_model_path', type=str, default=None)

################
# Dataset
################
parser.add_argument('--dataset_code', type=str, default='beauty', choices=DATASETS.keys())
parser.add_argument('--min_rating', type=int, default=0, help='Only keep ratings greater than equal to this value')
parser.add_argument('--min_uc', type=int, default=5, help='Only keep users with more than min_uc ratings')
parser.add_argument('--min_sc', type=int, default=5, help='Only keep items with more than min_sc ratings')
parser.add_argument('--split', type=str, default='leave_one_out', help='How to split the datasets')
parser.add_argument('--dataset_split_seed', type=int, default=98765)
parser.add_argument('--eval_set_size', type=int, default=256)

################
# Dataloader
################
parser.add_argument('--dataloader_code', type=str, default='bert', choices=DATALOADERS.keys())
parser.add_argument('--dataloader_random_seed', type=float, default=0.0)
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=64)

################
# NegativeSampler not used
################
parser.add_argument('--train_negative_sampler_code', type=str, default='random', choices=['popular', 'random'],
                    help='Method to sample negative items for training. Not used in bert')
parser.add_argument('--train_negative_sample_size', type=int, default=100)
parser.add_argument('--train_negative_sampling_seed', type=int, default=None)
parser.add_argument('--test_negative_sampler_code', type=str, default='random', choices=['popular', 'random'],
                    help='Method to sample negative items for evaluation')
parser.add_argument('--test_negative_sample_size', type=int, default=100)
parser.add_argument('--test_negative_sampling_seed', type=int, default=None)

################
# Trainer
################
parser.add_argument('--trainer_code', type=str, default='bert', choices=TRAINERS.keys())
# device #
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1)

parser.add_argument('--device_idx', type=str, default='0')
# optimizer #
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'])
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='l2 regularization')
parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
# lr scheduler #
parser.add_argument('--decay_step', type=int, default=15, help='Decay step for StepLR')
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR')
# epochs #
parser.add_argument('--num_epochs', type=int, default=250, help='Number of epochs for training')
# logger #
parser.add_argument('--log_period_as_iter', type=int, default=12800)
# evaluation #
parser.add_argument('--metric_ks', nargs='+', type=int, default=[1, 5, 10, 20], help='ks for Metric@k')
parser.add_argument('--best_metric', type=str, default='NDCG@10', help='Metric for determining the best model')

################
# Model
################
parser.add_argument('--model_code', type=str, default='bert', choices=MODELS.keys())
parser.add_argument('--model_init_seed', type=int, default=None)
# BERT #
parser.add_argument('--bert_max_len', type=int, default=None, help='Length of sequence for bert')
parser.add_argument('--bert_num_items', type=int, default=None, help='Number of total items')
parser.add_argument('--bert_hidden_units', type=int, default=None, help='Size of hidden vectors (d_model)')
parser.add_argument('--bert_num_blocks', type=int, default=None, help='Number of transformer layers')
parser.add_argument('--bert_num_heads', type=int, default=None, help='Number of heads for multi-attention')
parser.add_argument('--bert_dropout', type=float, default=None, help='Dropout probability to use throughout the model')
parser.add_argument('--bert_mask_prob', type=float, default=None, help='Probability for masking items in the training sequence')


parser.add_argument('--projectionhead', action='store_true')
parser.add_argument('--calcsim', type=str, default='cosine', choices=['cosine', 'dot'])



parser.add_argument('--alpha', type=float, default=0.1, help='loss proportion learning rate')
parser.add_argument('--lambda_', type=float, default=5, help='loss proportion significance indicator')
parser.add_argument('--tau', type=float, default=1, help='contrastive loss temperature')
parser.add_argument('--num_positive', type=int, default=4, help='number of positive samples')


###data augmentations###not used###
parser.add_argument('--augment_threshold', default=12, type=int, \
                    help="control augmentations on short and long sequences.\
                    default:-1, means all augmentations types are allowed for all sequences.\
                    For sequence length < augment_threshold: Insert, and Substitute methods are allowed \
                    For sequence length > augment_threshold: Crop, Reorder, Substitute, and Mask \
                    are allowed.")
parser.add_argument('--similarity_model_name', default='ItemCF_IUF', type=str, \
                    help="Method to generate item similarity score. choices: \
                    Random, ItemCF, ItemCF_IUF(Inverse user frequency), Item2Vec, LightGCN")
parser.add_argument("--augmentation_warm_up_epoches", type=float, default=80, \
                    help="number of epochs to switch from \
                    memory-based similarity model to \
                    hybrid similarity model.")
parser.add_argument('--base_augment_type', default='random', type=str, \
                    help="default data augmentation types. Chosen from: \
                    mask, crop, reorder, substitute, insert, random, \
                    combinatorial_enumerate (for multi-view).")
parser.add_argument('--augment_type_for_short', default='SIM', type=str, \
                    help="data augmentation types for short sequences. Chosen from: \
                    SI, SIM, SIR, SIC, SIMR, SIMC, SIRC, SIMRC.")
parser.add_argument("--truncation_rate", type=float, default=0.01)
parser.add_argument("--mask_rate", type=float, default=0.7, help="mask ratio for mask operator")
parser.add_argument("--reorder_rate", type=float, default=0.01, help="reorder ratio for reorder operator")
parser.add_argument("--substitute_rate", type=float, default=0.05, \
                    help="substitute ratio for substitute operator")
parser.add_argument("--insert_rate", type=float, default=0.5, \
                    help="insert ratio for insert operator")
parser.add_argument("--max_insert_num_per_pos", type=int, default=1, \
                    help="maximum insert items per position for insert operator - not studied")
#parser.add_argument('--n_views', type=int, default=2)

parser.add_argument('--DA_epochs', type=int, default=999, help='do not use data augmentation')

################

parser.add_argument('--validateafter', type=int, default=100, help='validate after some epochs - save time')

################
# Experiment
################
parser.add_argument('--experiment_dir', type=str, default='experiments')
parser.add_argument('--experiment_description', type=str, default='test')


################
args = parser.parse_args()

set_template(args)
