import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from wandb.sdk.wandb_run import Run

class CNNTrainer():
    def __init__(self, run:Run, model:nn.Module,train_dataloader:torch.utils.data.DataLoader, test_dataloader:torch.utils.data.DataLoader) -> None:
        self.run = run
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model

        self.lr = run.config['lr']

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(params=model.parameters(recurse=True),lr=self.lr)

    def _train_loop(self):
        self.model.train() #training mode 
        for i, (X, y) in enumerate(self.train_dataloader):
            preds = self.model(X) # make predictions
            loss = self.loss_fn(preds,y) #calculate loss with loss function
            loss.backward()

            pred_probabilities = torch.softmax(preds)
            most_probable = torch.argmax(pred_probabilities,1) #select highest probability of each class 
            num_correct = (most_probable == y).sum().item()
            accuracy = num_correct/(y.size(0))
            
            self.optimizer.step() #update params
            self.optimizer.zero_grad() #reset gradients for next pass
            self.run.log({ #log results
                "train_step":i+1,
                "train_CO_loss":loss,
                "train_accuracy":accuracy
            })
    
    def _test_loop(self):
        self.model.eval()    
        with torch.no_grad():
            for i, (X,y) in enumerate(self.test_dataloader):
                preds = self.model(X)
                loss = self.loss_fn(preds,y)

                pred_probabilities = torch.softmax(preds)
                most_probable = torch.argmax(pred_probabilities,1) #select highest probability of each class 
                num_correct = (most_probable == y).sum().item()
                accuracy = num_correct/(y.size(0))

                self.run.log({
                    "test_step":i+1,
                    "test_CO_loss":loss,
                    "test_accuracy":accuracy
                })

    def full_epoch_loop(self):
        '''
        Run train & test loop
        '''
        for i in range(self.run['epochs']):
            self._train_loop()
            self._test_loop()

    def train_epoch_loop(self):
        '''
        Run only train loop
        '''
        for i in range(self.run['epochs']):
            self._train_loop()

    def train_epoch_loop(self):
        '''
        Run only test loop
        '''
        for i in range(self.run['epochs']):       
            self._test_loop()
            

            
