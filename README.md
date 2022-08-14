# CBiT
Implementation of the paper "Contrastive Learning with Bidirectional Transformers for Sequential Recommendation".

## Run beauty
```
python main.py --template train_bert --dataset_code beauty
```

## Run toys
```
python main.py --template train_bert --dataset_code toys
```

## Run ml-1m
```
python main.py --template train_bert --dataset_code ml-1m
```

## Test a pretrained checkpoint on beauty
```
python test.py --template test_bert
```
* The checkpoint file is stored at /experiments/test_2022-04-26_0/models/best_acc_model.pth . You may need to download the checkpoint file manually from git LFS.

## Acknowledgements
Training pipeline is implemented based on this repo https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch . We would like to thank the contributors for their work.

## Citation
Please cite our paper if you find our codes useful:

```
@inproceedings{CBiT,
  author    = {Hanwen Du and
               Hui Shi and
               Pengpeng Zhao and
               Deqing Wang and
               Victor S. Sheng and
               Yanchi Liu and
               Guanfeng Liu and
               Lei Zhao},
  title     = {Contrastive Learning with Bidirectional Transformers for Sequential Recommendation},
  booktitle = {CIKM},
  year      = {2022}
}
```
