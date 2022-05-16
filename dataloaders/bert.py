from .base import AbstractDataloader
from .negative_samplers import negative_sampler_factory
from .data_augmentation import *
from .similarity import *

import torch
import torch.utils.data as data_utils
import copy


class BertDataloader(AbstractDataloader):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)

        args.num_items = len(self.smap)
        print("num_items:%d"%args.num_items)

        #args.similarity_model_path = os.path.join('./Data',
        #                                          args.dataset_code + '_' + args.similarity_model_name + '_similarity.pkl')

        #args.offline_similarity_model = OfflineItemSimilarity(train_data_dict=self.train,similarity_path=args.similarity_model_path,model_name=args.similarity_model_name)

        # -----------   online based on shared item embedding for item similarity --------- #

        #args.online_similarity_model = OnlineItemSimilarity(item_size=args.num_items + 2)

        self.max_len = args.bert_max_len

        self.mask_prob = args.bert_mask_prob

        self.CLOZE_MASK_TOKEN = self.item_count + 1
        self.train_slidewindow, self.train_slidewindow_by_user, self.user_count_slidewindow = self.get_train_dataset_slidewindow(args.slide_window_step)
        #code = args.train_negative_sampler_code
        # train_negative_sample_size=100  train_negative_sampling_seed=None
        #
        #train_negative_sampler = negative_sampler_factory(code, self.train_slidewindow_by_user, self.val, self.test,
        #                                                  self.user_count, self.item_count,
        #                                                  args.train_negative_sample_size,
        #                                                  args.train_negative_sampling_seed,
        #                                                  self.save_folder)
        #code = args.test_negative_sampler_code
        # test_negative_sample_size=100  test_negative_sampling_seed=None
        #test_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
        #                                                 self.user_count, self.item_count,
        #                                                 args.test_negative_sample_size,
        #                                                 args.test_negative_sampling_seed,
        #                                                 self.save_folder)

        #self.train_negative_samples = train_negative_sampler.get_negative_samples(sample_type="train")
        #self.test_negative_samples = test_negative_sampler.get_negative_samples(sample_type="test")
        #self.cl_data = RecWithContrastiveLearning(args,self.CLOZE_MASK_TOKEN)

        self.num_positive = args.num_positive

    def get_train_dataset_slidewindow(self, step=10):
        real_user_count=0
        train_slidewindow={}
        train_slidewindow_by_user = {}
        for user in range(self.user_count):
            if isinstance(self.train[user][1], tuple):
                seq = [x[0] for x in self.train[user]]
            else:
                seq = self.train[user]
            seq_len = len(seq)
            beg_idx = list(range(seq_len-self.args.bert_max_len, 0, -step))
            beg_idx.append(0)
            for i in beg_idx:

                temp = seq[i:i + self.args.bert_max_len]
                train_slidewindow[real_user_count] = temp

                l = train_slidewindow_by_user.get(user,[])
                l.append(temp)
                train_slidewindow_by_user[user] = l

                real_user_count+=1
            '''
            all_documents[user] = [
                item_seq[i:i + max_num_tokens] for i in beg_idx[::-1]
            ]
            '''
        return train_slidewindow, train_slidewindow_by_user, real_user_count
    @classmethod
    def code(cls):
        return 'bert'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()

        #cl_loader = data_utils.DataLoader(BertCLDataset(self.train,self.cl_data), batch_size=self.args.train_batch_size,shuffle=True, pin_memory=True)
        #finetune_loader = data_utils.DataLoader(BertFinetuneDataset(self.train, self.val, self.max_len, self.CLOZE_MASK_TOKEN), batch_size=self.args.train_batch_size,shuffle=True, pin_memory=True)

        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True)
        return dataloader

    def _get_train_dataset(self):
        print("users:%d"%len(self.train))
        print("pseudo users:%d"%len(self.train_slidewindow))
        dataset = BertTrainDataset(self.train_slidewindow,self.num_positive,self.max_len, self.mask_prob, self.CLOZE_MASK_TOKEN, self.item_count, self.rng)
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):

        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size

        dataset = self._get_eval_dataset(mode)

        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, pin_memory=True)
        return dataloader

    def _get_eval_dataset(self, mode):
        if mode=='val':
            answers = self.val
            dataset = BertEvalDataset(self.train, answers, self.max_len, self.item_count, self.CLOZE_MASK_TOKEN)
        else:
            answers = self.test
            dataset = BertTestDataset(self.train, self.val, answers, self.max_len, self.item_count, self.CLOZE_MASK_TOKEN)
        return dataset

class RecWithContrastiveLearning():
    def __init__(self, args, MASK_TOKEN):
        self.args = args
        self.max_len = args.bert_max_len
        # currently apply one transform, will extend to multiples
        # it takes one sequence of items as input, and apply augmentation operation to get another sequence
        self.similarity_model = [args.offline_similarity_model, args.online_similarity_model]
        self.augmentations = {'truncation': Truncation(truncation_rate=args.truncation_rate),
                              'mask': Mask(gamma=args.mask_rate,mask_token=MASK_TOKEN),
                              'reorder': Reorder(beta=args.reorder_rate),
                              'substitute': Substitute(self.similarity_model,
                                                substitute_rate=args.substitute_rate),
                              'insert': Insert(self.similarity_model,
                                               insert_rate=args.insert_rate,
                                               max_insert_num_per_pos=args.max_insert_num_per_pos),
                              'random': Random(truncation_rate=args.truncation_rate, gamma=args.mask_rate,
                                                beta=args.reorder_rate, item_similarity_model=self.similarity_model,
                                                insert_rate=args.insert_rate,
                                                max_insert_num_per_pos=args.max_insert_num_per_pos,
                                                substitute_rate=args.substitute_rate,
                                                augment_threshold=self.args.augment_threshold,
                                                augment_type_for_short=self.args.augment_type_for_short),
                              'combinatorial_enumerate': CombinatorialEnumerate(truncation_rate=args.truncation_rate, gamma=args.mask_rate,
                                                beta=args.reorder_rate, item_similarity_model=self.similarity_model,
                                                insert_rate=args.insert_rate,
                                                max_insert_num_per_pos=args.max_insert_num_per_pos,
                                                substitute_rate=args.substitute_rate, n_views=args.n_views)
                            }
        if self.args.base_augment_type not in self.augmentations:
            raise ValueError(f"augmentation type: '{self.args.base_augment_type}' is invalided")
        print(f"Creating Contrastive Learning Dataset using '{self.args.base_augment_type}' data augmentation")
        self.base_transform = self.augmentations[self.args.base_augment_type]

    def augment(self, input_ids):
        augmented_seqs = []
        for i in range(2):
            augmented_input_ids = self.base_transform(input_ids)
            pad_len = self.max_len - len(augmented_input_ids)
            augmented_input_ids = [0] * pad_len + augmented_input_ids

            augmented_input_ids = augmented_input_ids[-self.max_len:]

            augmented_seqs.append(augmented_input_ids)
        return augmented_seqs

class BertTrainDataset(data_utils.Dataset):

    def __init__(self, u2seq, num_positive, max_len, mask_prob, mask_token, num_items, rng):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.num_positive = num_positive
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = rng



    def __len__(self):
        return len(self.users)
    def get_masked_seq(self, seq):
        tokens = []
        labels = []

        for s in seq:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:

                    tokens.append(self.mask_token)

                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.num_items))

                else:
                    tokens.append(s)

                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)

            tokens = tokens[-self.max_len:]
            labels = labels[-self.max_len:]

            mask_len = self.max_len - len(tokens)

            tokens = [0] * mask_len + tokens
            labels = [0] * mask_len + labels
        return tokens,labels
    def __getitem__(self, index):
        user = self.users[index]
        seq = self._getseq(user)
        #negs = self.negative_samples[user]


        '''
        for s in seq:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:

                    tokens.append(self.mask_token)

                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.num_items))

                else:
                    tokens.append(s)

                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)



        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]
        negs = negs[-self.max_len:]

        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        '''
        return_list=[]
        for i in range(self.num_positive):
            tokens, labels = self.get_masked_seq(seq)
            return_list.append(torch.LongTensor(tokens))
            return_list.append(torch.LongTensor(labels))
        #mask_len = self.max_len - len(tokens)
        #negs = [0] * mask_len + negs
        #return_list.append(torch.LongTensor(negs))
        return tuple(return_list)

        #return torch.LongTensor(tokens),  torch.LongTensor(labels), torch.LongTensor(negs)

    def _getseq(self, user):
        return self.u2seq[user]

class BertFinetuneDataset(data_utils.Dataset):

    def __init__(self, u2seq, label, max_len, mask_token):

        self.u2seq = u2seq

        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len

        self.mask_token = mask_token
        self.label = label

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self._getseq(user)
        negs = self.negative_samples[user]

        tokens = seq+[self.mask_token]
        labels = self.label[user]
        tokens = tokens[-self.max_len:]
        negs = negs[-self.max_len:]

        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        labels = [0] * (self.max_len-1) + labels
        negs = [0] * (self.max_len-len(negs)) + negs

        return torch.LongTensor(tokens),  torch.LongTensor(labels), torch.LongTensor(negs)

    def _getseq(self, user):
        return self.u2seq[user]

class BertCLDataset(data_utils.Dataset):

    def __init__(self, u2seq,cl_data):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.cl_data=cl_data


    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):

        user = self.users[index]
        seq = self._getseq(user)
        aug = self.cl_data.augment(seq)

        return torch.LongTensor(aug[0]), torch.LongTensor(aug[1])

    def _getseq(self, user):
        return self.u2seq[user]


class BertEvalDataset(data_utils.Dataset):
    # self.train, answers, self.max_len, self.CLOZE_MASK_TOKEN, self.test_negative_samples
    def __init__(self, u2seq, u2answer, max_len, num_items,mask_token):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.num_items = num_items
        self.mask_token = mask_token

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):

        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]

        seq = seq + [self.mask_token]

        seq = seq[-self.max_len:]

        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        interacted = set(answer+seq)# remove items that user has interacted with as candidate items
        candidates = answer + [x for x in range(1,self.num_items+1) if x not in interacted]
        candidates = candidates + [0]*(self.num_items-len(candidates))#rank on the whole item set

        labels = [1] * len(answer) + [0] * (len(candidates)-1)

        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)

class BertTestDataset(data_utils.Dataset):
    # self.train, answers, self.max_len, self.CLOZE_MASK_TOKEN, self.test_negative_samples
    def __init__(self, u2seq, u2val, u2answer, max_len, num_items, mask_token):
        self.u2seq = u2seq
        self.u2val = u2val
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.num_items = num_items
        self.mask_token = mask_token

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):

        user = self.users[index]
        seq = self.u2seq[user]
        val = self.u2val[user]
        answer = self.u2answer[user]


        #candidates = answer + [x for x in range(1,self.num_items+1) if x!=answer[0]]
        '''
        for i in range(len(candidates)):
            if candidates[i] in seq or candidates[i] in val:
                candidates[i]=0
        '''

        #labels = [1] * len(answer) + [0] * (len(candidates)-1)


        seq = seq + val + [self.mask_token]

        seq = seq[-self.max_len:]

        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        interacted = set(answer+seq)# remove items that user has interacted with as candidate items
        candidates = answer + [x for x in range(1,self.num_items+1) if x not in interacted]#rank on the whole item set
        candidates = candidates + [0]*(self.num_items-len(candidates))

        labels = [1] * len(answer) + [0] * (len(candidates)-1)


        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)
