from datasets import dataset_factory
from .bert import BertDataloader
from .ae import AEDataloader


DATALOADERS = {
    # "bert"
    BertDataloader.code(): BertDataloader,
    AEDataloader.code(): AEDataloader
}


def dataloader_factory(args):
    # 调用datasets文件夹中__init__.py文件中的dataset_factory函数
    # <class 'datasets.ml_1m.ML1MDataset'>
    # 选择数据集
    dataset = dataset_factory(args)
    # <class 'dataloaders.bert.BertDataloader'>
    dataloader = DATALOADERS[args.dataloader_code]
    # 实例化类
    # <dataloaders.bert.BertDataloader object at 0x7ff8e0973748>
    # 得到负样本
    dataloader = dataloader(args, dataset)
    # 调用get_pytorch_dataloaders函数
    train, val, test = dataloader.get_pytorch_dataloaders()
    return train, val, test
