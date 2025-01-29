import json
from torch import Generator


def read_config(config_path) -> dict: #load config dictionary
        with open(config_path) as conf_file:
            config_dict = json.load(conf_file)
            return config_dict 
        

if __name__ == '__main__':
    print('starting training - Cifar-10')
    import torch.utils
    from torch.utils.data import RandomSampler
    from torchvision.transforms import v2
    import json
    from torch.utils.data import DataLoader, Subset
    from modelC3 import CNN
    import torch
    import wandb
    from trainer_def import CNNTrainer
    from init_weights import init_model_w
    from torchvision.datasets import CIFAR10
    
    dataset_config = read_config('experiment/datasets_cifar10.json')
    set_prefix = 'C'
    n_sets = 11
    sets_start = 1
    sets_end = 11

    if sets_end > n_sets:
        raise ValueError('sets_end')

    for dataset_n in range(sets_start,sets_end+1):
        data_group  = set_prefix+str(dataset_n)
        
        config_dict = read_config('experiment/config.json')
        config_dict['data_group'] = data_group
        config_dict['train_size'] = dataset_config[data_group]['train_size']
        config_dict['epochs'] = dataset_config[data_group]['epochs']
        config_dict['base_epochs'] = dataset_config[data_group]['base_epochs']

        device = ( #selects device
                'cuda'
                if torch.cuda.is_available()
                else 'mps'
                if torch.backends.mps.is_available()
                else 'cpu'
            )
        print(device)
        n_runs = 10
        start_n=0

        #define transform to augment image
        image_transform = v2.Compose([
            v2.Resize(32),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]), # to tensor
            v2.RandomHorizontalFlip(p=1), #100% probability
            v2.RandomRotation(degrees=(90,90)), #flip 90 degrees
            v2.Normalize([0.491, 0.482, 0.446],[0.247, 0.243, 0.261]) # https://stackoverflow.com/questions/66678052/how-to-calculate-the-mean-and-the-std-of-cifar10-data/69699979#69699979
        ])
        train_dataset = CIFAR10('torch_Cifar10',
            train=True,
            download=True,
            transform = image_transform,)
        print(train_dataset.__len__())
        test_dataset = CIFAR10('torch_Cifar10',
            train=False,
            download=True,
            transform = image_transform,)
        
        print(test_dataset.__len__())

        for i in range(start_n,n_runs):
            #set random seeds for reproducibility
            seed = 100*i + i
            print(f'---------- Run {seed} ----------')
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.mps.manual_seed(seed)

            train_subset_size = config_dict['train_size']
            #init wandb run
            run = wandb.init(project='GA-Letters',name = f"Cifar-D{train_subset_size}-S{seed}", config=config_dict,job_type='experimental run',group=config_dict['data_group'])
            test_subset_size = config_dict['test_size']
            #subset train data


            #subset test data:
            batch_size = run.config['batch_size']

            n_workers = 4
            #dataloaders
            r_generator = Generator().manual_seed(seed)
            r_sampler = RandomSampler(train_dataset,True,num_samples=train_subset_size,generator=r_generator)

            train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size, num_workers=n_workers,sampler=r_sampler)
            test_dataloader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True, num_workers=n_workers)
            loader_size = len(train_dataloader)

            test_ex = train_dataset.__getitem__(0)

            #get hyperparams from config
            in_channels = run.config['input_channels']
            conv1 = run.config['conv1']
            ckernel1 = run.config['ckernel1']
            MPkernel1 = run.config['MPkernel1']
            conv2 = run.config['conv2']
            ckernel2 = run.config['ckernel2']
            MPkernel2 = run.config['MPkernel2']
            conv3 = run.config['conv3']
            ckernel3 = run.config['ckernel3']
            MPkernel3 = run.config['MPkernel3']
            fc1 = run.config['fc1']

            model = CNN(
                input_channels=in_channels,
                conv1=conv1,
                ckernel1=ckernel1,
                MPkernel1=MPkernel1,
                conv2=conv2,
                ckernel2=ckernel2,
                MPkernel2=MPkernel2,
                conv3=conv3,
                ckernel3=ckernel3,
                MPkernel3=MPkernel3,
                fc1=fc1,
                out_dim=26,
                input_HW=32,
                device=device
            )
            run.watch(models=model,criterion=torch.nn.functional.cross_entropy,log='all')
            #model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True) #resnet18
            model.apply(init_model_w)
            model.to(device=device)
            #init_model_w(model=model)
            
            #train
            trainer = CNNTrainer(run=run,
                                model = model,
                                device=device,
                                train_dataloader=train_dataloader,
                                test_dataloader=test_dataloader,
                                report_freq=200)
            trainer.full_epoch_loop(print_gradients=True,base_epochs=config_dict.get('base_epochs'))
            '''torch.save(trainer.model.state_dict(),f'models/S{seed}-{datetime.datetime.now()}.pt')
            writer = SummaryWriter(f'models/D5/S{seed}/',)
            writer_input_batch, _ = next(iter(train_dataloader))
            writer_input_batch:torch.Tensor 
            writer_input_batch = writer_input_batch.to(device=device)
            writer.add_graph(model=trainer.model,input_to_model=writer_input_batch)'''
            run.finish(0)
