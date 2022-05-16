
from options import args
from models import BERTModel
from datasets import dataset_factory
from dataloaders.bert import BertDataloader
from trainers.bert import BERTTrainer
from utils import *


def train():

    export_root = setup_train(args)

    dataset = dataset_factory(args)

    train_loader, val_loader, test_loader = BertDataloader(args, dataset).get_pytorch_dataloaders()

    model = BERTModel(args)

    trainer = BERTTrainer(args, model, train_loader, val_loader, test_loader, export_root)

    trainer.train()
    
    trainer.test()


if __name__ == '__main__':
    train()
