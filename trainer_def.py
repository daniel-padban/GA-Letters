import trace
from numpy import mean
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from wandb.sdk.wandb_run import Run

class CNNTrainer():
    def __init__(self, run:Run, model:nn.Module,device:str,train_dataloader:torch.utils.data.DataLoader, test_dataloader:torch.utils.data.DataLoader) -> None:
        self.run = run
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.device = device

        self.lr = run.config['lr']
        self.clip = run.config['clip']

        self.loss_fn = torch.nn.CrossEntropyLoss()
        #self.optimizer = torch.optim.AdamW(params=model.parameters(recurse=True),lr=self.lr)
        self.optimizer = torch.optim.SGD(params=model.parameters(),lr=self.lr,momentum=0.9)

    def _train_loop(self,epoch):
        self.model.train(True) #training mode 
        step_group = epoch*len(self.train_dataloader)  # calculates how many steps have been processed already, start for count, epoch starts at 0
        running_loss = 0
        for i, (X, y) in enumerate(self.train_dataloader):
            self.optimizer.zero_grad() #reset gradients for next pass
            X = X.repeat(1,3,1,1) #for resnet
            X = X.to(device=self.device)
            y = y.to(device=self.device)
            X = X.requires_grad_(True)
            preds = self.model(X) # make predictions
            loss = self.loss_fn(preds,y) #calculate loss with loss function
            current_loss = loss.item()
            running_loss += current_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip) #gradient clipping

            pred_probabilities = torch.softmax(preds,1)
            most_probable = torch.argmax(pred_probabilities,1) #select highest probability of each class 
            num_correct = (most_probable == y).sum().item()
            accuracy = num_correct/(y.size(0))
            
            self.optimizer.step() #update params
            self.run.log({ #log results
                "train_step":i+1+step_group,
                "train_CO_loss":loss,
                "train_accuracy":accuracy
            })
        mean_loss = running_loss/len(self.train_dataloader)
        return mean_loss
    
    def _test_loop(self,epoch):
        self.model.eval()    
        with torch.no_grad():
            step_group = epoch*len(self.test_dataloader) # calculates how many steps have been processed already, start for count, epoch starts at 0
            for i, (X,y) in enumerate(self.test_dataloader):
                X = X.repeat(1,3,1,1) #for resnet
                X = X.to(device=self.device)
                y = y.to(device=self.device)
                
                preds = self.model(X)
                loss = self.loss_fn(preds,y)

                pred_probabilities = torch.softmax(preds,1)
                most_probable = torch.argmax(pred_probabilities,1) #select highest probability of each class 
                num_correct = (most_probable == y).sum().item()
                accuracy = num_correct/(y.size(0))

                self.run.log({
                    "test_step":i+1+step_group,
                    "test_CO_loss":loss,
                    "test_accuracy":accuracy
                })

    def full_epoch_loop(self,print_gradients:bool=False):
        '''
        Run train & test loop
        '''
        for epoch in range(self.run.config['epochs']):
            print(f'Epoch:{epoch+1}')
            train_mean_loss = self._train_loop(epoch=epoch)
            self._test_loop(epoch=epoch)
            if print_gradients:
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        print(f'{name}: {param.grad.mean()}')

            self.run.log({"epoch":epoch+1,'train_mean_loss':train_mean_loss})

    def train_epoch_loop(self):
        '''
        Run only train loop
        '''
        for epoch in range(self.run.config['epochs']):
            self._train_loop(epoch)

    def train_epoch_loop(self):
        '''
        Run only test loop
        '''
        for epoch in range(self.run.config['epochs']):       
            self._test_loop(epoch)
            

            
