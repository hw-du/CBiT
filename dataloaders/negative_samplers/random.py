from .base import AbstractNegativeSampler

from tqdm import trange

import numpy as np


class RandomNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'random'

    def generate_negative_samples(self,sample_type='test'):
        assert self.seed is not None, 'Specify seed for random sampling'
        np.random.seed(self.seed)
        negative_samples = {}
        print('Sampling negative items')
        # tqdm包中的函数
        real_user_count=0
        for user in trange(self.user_count):
            # 记录数据集中已经出现的物品
            # if isinstance(self.train[user][1], tuple):
            #     print('I am in')
            #     seen = set(x[0] for x in self.train[user])
            #     sample_count = len(seen)
            #     seen.update(x[0] for x in self.val[user])
            #     seen.update(x[0] for x in self.test[user])
            # else:
            #only one document is available under the name of this user
            if isinstance(self.train[user][0], int):
                seqs=[self.train[user]]
            else:
                seqs=self.train[user]
            seen=set()
            for seq_ in seqs:
                seen.update(seq_)
            seen.update(self.val[user])
            seen.update(self.test[user])
            for seq in seqs:
                sample_count = len(seq)
                samples = []
                # 100
                if sample_type=="test":
                    sample_count=self.sample_size
                sample_count=min(sample_count,self.sample_size)
                for _ in range(sample_count):
                    item = np.random.choice(self.item_count) + 1
                    while item in seen or item in samples:
                        item = np.random.choice(self.item_count) + 1
                    samples.append(item)
                # 保存每条记录的负样本
                negative_samples[real_user_count] = samples
                real_user_count+=1
        return negative_samples

    def generate_negative_samples_by_user(self):
        assert self.seed is not None, 'Specify seed for random sampling'
        np.random.seed(self.seed)
        negative_samples = {}
        print('Sampling negative items')
        # tqdm包中的函数
        real_user_count=0
        for user in trange(self.user_count):
            for seq in self.train[user]:
            # 记录数据集中已经出现的物品
                if isinstance(seq[1], tuple):
                    seen = set(x[0] for x in seq)
                    sample_count = len(seq)
                    seen.update(x[0] for x in self.val[user])
                    seen.update(x[0] for x in self.test[user])
                else:
                    seen = set(seq)
                    sample_count = len(seq)
                    seen.update(self.val[user])
                    seen.update(self.test[user])

                samples = []
                # 100
                sample_count = min(sample_count, self.sample_size)
                for _ in range(sample_count):
                    item = np.random.choice(self.item_count) + 1
                    while item in seen or item in samples:
                        item = np.random.choice(self.item_count) + 1
                    samples.append(item)
                # 保存每条记录的负样本
                negative_samples[real_user_count] = samples
                real_user_count+=1

        return negative_samples
