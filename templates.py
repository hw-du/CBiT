def set_template(args):
    if args.template is None:
        return

    else:
        args.mode = 'train'
        #args.dataset_code = 'beauty'
        args.min_rating = 0
        args.min_uc = 5
        args.min_sc = 5
        args.split = 'leave_one_out'

        args.dataloader_code = 'bert'
        batch = 256
        if args.dataset_code=='beauty':
            args.bert_dropout = 0.3
            args.tau = 0.3
            seq_len = 15
            args.num_positive = 4
            args.decay_step = 100
        elif args.dataset_code=='toys':
            args.decay_step = 100
            args.bert_dropout = 0.3
            args.tau = 0.3
            args.num_positive = 5
            seq_len = 20
        elif 'ml' in args.dataset_code:
            args.decay_step = 50
            seq_len = 40
            args.tau = 0.3
            args.num_positive = 8
            args.bert_dropout = 0.2
        else:
            args.decay_step = 50
            seq_len = 20
            args.bert_dropout = 0.1
     
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch

        #args.train_negative_sampler_code = 'random'

        args.train_negative_sample_size = seq_len

        #args.train_negative_sampling_seed = 56789
        #args.test_negative_sampler_code = 'random'
        #args.test_negative_sample_size = seq_len
        #args.test_negative_sampling_seed = 98765

        args.trainer_code = 'bert'
        #args.device = 'cuda:2'
        args.num_gpu = 1
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
   
        args.gamma = 1.0

        args.num_epochs = 250
        args.metric_ks = [1, 5, 10, 20]
        args.best_metric = 'NDCG@10'

        args.model_code = 'bert'
        args.model_init_seed = 0

        args.model_sample_seed=0
        
        args.bert_hidden_units = 256
        args.bert_mask_prob = 0.15
        args.bert_max_len = seq_len
        args.bert_num_blocks = 2
        args.bert_num_heads = 4 if 'ml' in args.dataset_code else 2

        args.slide_window_step = 10 if 'ml' in args.dataset_code else 1




