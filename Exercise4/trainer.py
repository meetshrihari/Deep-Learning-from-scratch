import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
#from evaluation import create_evaluation

class Trainer:
    
    def __init__(self,               
                 model,                # Model to be trained.
                 crit,                 # Loss function
                 optim = None,         # Optimiser
                 train_dl = None,      # Training data set
                 val_test_dl = None,   # Validation (or test) data set
                 cuda = True,          # Whether to use the GPU
                 early_stopping_cb = None): # The stopping criterion. 
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self._early_stopping_cb = early_stopping_cb



        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss
        #TODO
        self._optim.zero_grad()
        prediction = self._model(x)
        loss = self._crit(prediction, t.squeeze(y.float()))
        loss.backward()
        self._optim.step()
        return loss.item()
        
    
    def val_test_step(self, x, y):
        
        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions
        #TODO
        prediction = self._model(x)
        loss = self._crit(prediction, t.squeeze(y.float()))
        return loss.item(), prediction
        
    def train_epoch(self):
        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it
        #TODO

        self._model = self._model.train()
        loss = 0
        for img, labels in self._train_dl: #consider this looping over batches one by one, set one batch length later
            if self._cuda:
                img = img.to('cuda')
                labels = labels.to('cuda')
            loss += self.train_step(img, labels)
        avg_loss = loss/8
        return avg_loss



    
    def val_test(self):
        # set eval mode
        # disable gradient computation
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        #TODO
        self._model = self._model.eval()
        total_loss = 0
        self.f1score= 0
        with t.no_grad():
            for images, labels in self._val_test_dl:
                if self._cuda:
                    images = images.to('cuda')
                    labels = labels.to('cuda')
                loss, pred = self.val_test_step(images, labels)
                self.f1score += f1_score(t.squeeze(labels.cpu()), t.squeeze(t.nn.functional.sigmoid(pred.cpu()).round()), average= None)
                total_loss += loss
        print('f1score = ' + str(self.f1score/2))
        return total_loss/2
    
    def fit(self, epochs=-1):
        assert self._early_stopping_cb is not None or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        #TODO
        train_loss = []
        validation_loss = []
        epoch_counter = 1
        maxf1 = []


        while True:
            # stop by epoch number
            # train for a epoch and then calculate the loss and metrics on the validation set
            # append the losses to the respective lists 
            # use the save_checkpoint function to save the model for each epoch
            # check whether early stopping should be performed using the early stopping callback and stop if so
            # return the loss lists for both training and validation
            #TODO
            print(epoch_counter)

            if epoch_counter == epochs:
                break
            avglossepoch = self.train_epoch()
            avgvalloss = self.val_test()
            print(avglossepoch, avgvalloss)

            epof1 = self.f1score / 2
            avgf1 = (epof1[0] + epof1[1]) / 2
            maxf1.append(avgf1)

            train_loss.append(avglossepoch)
            validation_loss.append(avgvalloss)
            #self.save_checkpoint(epoch_counter)
            self._early_stopping_cb.step(avgvalloss)
            stop = self._early_stopping_cb.should_stop()
            if stop:
                #self.save_checkpoint(7)
                print('maxf1reached = ' + str(max(maxf1)) + ' at epoch: ' + str(maxf1.index(max(maxf1))))
                print(min(validation_loss))
                break
            epoch_counter +=1

        return train_loss, validation_loss

                    
        
        
        