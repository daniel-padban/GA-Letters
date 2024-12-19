import json
def subtract_one(y):
        result = y-1
        return result

def read_config(config_path) -> dict: #load config dictionary
        with open(config_path) as conf_file:
            config_dict = json.load(conf_file)
            return config_dict 
        
print('starting training')
if __name__ == '__main__':
    dataset_config = read_config('datasets.json')
    set_prefix = 'B'
    n_sets = 11
    sets_start = 11
    sets_end = 7

    if sets_end > n_sets:
        raise ValueError('sets_ end')

    for dataset_n in range(sets_start,sets_end+1):
        data_group  = set_prefix+str(dataset_n)
        
        config_dict = read_config('config.json')
        config_dict['data_group'] = data_group
        config_dict['train_size'] = dataset_config[data_group]['train_size']
        config_dict['epochs'] = dataset_config[data_group]['epochs']
        config_dict['base_epochs'] = dataset_config[data_group]['base_epochs']


        import torch.utils
        import torch.utils.data
        from torchvision.transforms import v2
        import json
        from torch.utils.data import DataLoader, Subset
        from modelC3 import CNN
        import torch
        import wandb
        from trainer_def import CNNTrainer
        from init_weights import init_model_w
        from torchvision.datasets import EMNIST
        import datetime
        from torch.utils.tensorboard import SummaryWriter
        

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
            v2.Resize(28),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]), # to tensor
            v2.RandomHorizontalFlip(p=1), #100% probability
            v2.RandomRotation(degrees=(90,90)), #flip 90 degrees
            v2.Normalize((0.1736,),(0.3248,))
        ])
        target_transform = subtract_one #labels are shifted -1 step, range changes from 1-26 to 0-25, (needed for loss calculation)
        full_train_dataset = EMNIST('torch_EMNIST',
            split='letters',
            train=True,
            download=True,
            transform = image_transform,
            target_transform = target_transform)
        print(full_train_dataset.__len__())
        full_test_dataset = EMNIST('torch_EMNIST',
            split='letters',
            train=False,
            download=True,
            transform = image_transform,
            target_transform = target_transform)
        print(full_test_dataset.__len__())

        for i in range(start_n,n_runs):
            #set random seeds for reproducibility
            seed = 100*i + i
            print(f'---------- Run {seed} ----------')
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.mps.manual_seed(seed)

            train_subset_size = config_dict['train_size']
            #init wandb run
            run = wandb.init(name = f"Run-D{train_subset_size}-S{seed}", config=config_dict,job_type='experimental run',group=config_dict['data_group'])
            test_subset_size = config_dict['test_size']
            #subset train data
            train_indices = range(0,train_subset_size)
            train_dataset = Subset(full_train_dataset,train_indices)
            #subset test data:
            test_indices = range(0,test_subset_size)
            test_dataset = Subset(full_test_dataset,test_indices)
            batch_size = run.config['batch_size']

            n_workers = 4
            #dataloaders
            train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True, num_workers=n_workers)
            test_dataloader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True, num_workers=n_workers)

            #get hyperparams from config
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
                input_channels=1,
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
                input_HW=28,
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
