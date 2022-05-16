
from options import args
from models import BERTModel
from datasets import dataset_factory
from dataloaders.bert import BertDataloader
from trainers.bert import BERTTrainer
from utils import *

def train():
    export_root = setup_test(args)
    dataset = dataset_factory(args)
    train_loader, val_loader, test_loader = BertDataloader(args, dataset).get_pytorch_dataloaders()
    model = BERTModel(args)
    trainer = BERTTrainer(args, model, train_loader, val_loader, test_loader, export_root)
    best_model = torch.load(os.path.join('./experiments/test_2022-04-26_0/', 'models', 'best_acc_model.pth')).get('model_state_dict')
    model.load_state_dict(best_model)
    trainer.test()


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    else:
        raise ValueError('Invalid mode')
