'''
Project done in the framework of the Statistical Counsulting course for the SWDE (La Société Walonne Des Eaux) 
during the academic year 2023-2024.
'''

#%% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skmultilearn.model_selection import iterative_train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import  AutoTokenizer


from time import time
from statistics import variance
from sklearn.preprocessing import StandardScaler
#%% Import the dataset

path = 'C:/Users/User/Documents/Data Science/Xx_M1_5/Q2/LSTAT2380 Statistical Consulting/Project_SWDE/Data/finalDS_stemmed.csv'
fullDS = pd.read_csv(path, index_col = 0)
fullDS['Class'].value_counts()

#--- merge class 1 into the 3th
fullDS['Class'] = fullDS['Class'].replace({1:3})

#---  Remove the points of the class 3
fullDS = fullDS[fullDS['Class'] != 3]

#--- Reset index
fullDS.reset_index(drop=True, inplace=True)

# set variable to categorical
to_cat = ['Type_work','Equipe','Div. planif.','Localité',"Commune","FctCalculee","Class","Year_st","Month_st","PRISE_EAU","HAUTE_TENSION","STOKAGE_EAU","TRAITEMENT","EEM"]
fullDS[to_cat] = fullDS[to_cat].astype("category")

fullDS.dtypes
variance(fullDS['Volume_trav'])
variance(fullDS['ID OT Complet'])
variance(fullDS["ID OT short - pas d'unicité !"])

# scale the numerical variabless
to_scale = ['Volume_trav','ID OT Complet',"ID OT short - pas d'unicité !"]
scaler = StandardScaler()
for i in to_scale:
    fullDS[i] = scaler.fit_transform(fullDS[[i]])
    
del(to_cat, to_scale,i, scaler)


#%% PREPARE THE DATA FORMAT
#---
#final_text = tuple(final_text)

# --- Select the text
Y = fullDS['Class']
corres_lab,Y = np.unique(fullDS['Class'], return_inverse=True)
rows_idx = np.arange(len(Y))
len(rows_idx)
np.unique(Y)

#--- prepare Train/test with equivalent proportion of classes

Xtr_idx,y_train, Xte_idx, y_test = iterative_train_test_split(rows_idx[:,np.newaxis], 
                                                              Y[:,np.newaxis], test_size=0.15) # keep the proportion within the sets

x_train = [fullDS.loc[i,'Cleaned'] for i in Xtr_idx.flatten()]
x_test = [fullDS.loc[i,'Cleaned'] for i in Xte_idx.flatten()]
del(Xte_idx,Xtr_idx)

#--- Instantiate Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')

'''
#%% TRYING OVERSAMPLING
from imblearn.over_sampling import RandomOverSampler

# now oversample
oversampler = RandomOverSampler()
x_train_reshaped =  np.array(x_train).reshape(-1, 1)
x_train_re1, y_train_re= oversampler.fit_resample(x_train_reshaped, y_train)


#%%
'''
#--- Tokenize train/test sets
x_train_tok = tokenizer(x_train, padding = True, max_length  = 32, # ??? originaly unspecified value of max_length
                 truncation = True,return_tensors = 'pt').data['input_ids']
x_test_tok = tokenizer(x_test, padding = True, max_length  = 32 , # ??? same 
                 truncation = True,return_tensors = 'pt').data['input_ids'] 

#--- Get Y into a tensor format 
y_train_T = torch.Tensor(y_train.tolist()).long() 
y_test_T = torch.Tensor(y_test.tolist()).long()

#--- Convert data to torch dataset
class CrashDS(Dataset):
    
    def __init__(self,X,y):
        
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,idx):
        return self.X[idx],self.y[idx]

#--- Get train and test into Dataset class
train_d = CrashDS(X= x_train_tok,y = y_train_T)
test_d = CrashDS(X = x_test_tok, y = y_test_T)
#del(x_train,y_train,x_test,y_test)


#--- specify the way the data will be process => BATCH_SIZE 
train_loader = DataLoader(train_d, batch_size = 16,shuffle = True) # ???
test_loader = DataLoader(test_d,batch_size = 16, shuffle = True) # ???

#%% BUILD THE MODEL

config = transformers.DistilBertConfig(dropout = 0.15, acttention_dropout = 0.25) # initially 0.2 ???

#model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", config = config)
model = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased', config=config)
sample = x_train_tok[0:5]
print(f'Obj type {type(model(sample))}')
print(f'Output format : {model(sample)[0].shape} ')
print('Output used as input for the classifier (shape): ', model(sample)[0][:,0,:].shape)

#--- Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class DistilBertClassification(nn.Module):
    def __init__(self):
        super(DistilBertClassification, self).__init__()
        self.dbert = model
        self.dropout = nn.Dropout(p = 0.15) # ???
        self.linear1 = nn.Linear(768,64)
        self.ReLU = nn.ReLU()
        self.linear2 = nn.Linear(64,8)
        
    def forward(self, x):
        x = self.dbert(input_ids = x)
        x = x["last_hidden_state"][:, 0, :]
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.ReLU(x)
        logits = self.linear2(x)
        
        return logits

model_pt = DistilBertClassification().to(device) 
print(model_pt)

#--- Freeze BERT params 

for param in model_pt.dbert.parameters():
    param.requires_grad = False
del(param)

#--- Observe the number of parameters
tot_params = sum(p.numel() for p in model_pt.parameters()) # .numel() returns the totoal number of ele in the input tensor    
tot_trainable = sum(p.numel() for p in model_pt.parameters() if p.requires_grad)
print("Number of parameters: ", tot_params)
print("Number of trainable parameters: ", tot_trainable)
del(tot_params,tot_trainable)


#%% analyze proportion within x test and train

fullDS['Class'].value_counts()
corres_lab

u_val, counts = np.unique(y_test,return_counts= True)
val_counts_test = dict(zip(u_val,counts))

u_val, counts = np.unique(y_train,return_counts= True)
val_counts_train = dict(zip(u_val,counts))
print(val_counts_test)
print(val_counts_train)
print(corres_lab)
print(fullDS['Class'].value_counts())

#--- get a clear label mapping since they're transformed

#-- maps from old to new
label_map_old_to_new = {    # first is the old class label    # second number is the new class label
        0:0,
        2:1,
        4:2,
        5:3,
        6:4,
        7:5,
        8:6,
        9:7}



#--- Compute class's weights

#- Using complement of the class frequencies
class_weights = 1 - (fullDS['Class'].value_counts()/len(fullDS))
class_weights = dict(class_weights)
class_weights1 = {label_map_old_to_new[k]: v for k, v in class_weights.items()}


#- Using Inverse Frequence weightings
class_weights2 = 1/fullDS['Class'].value_counts()
class_weights2 = class_weights2/ class_weights2.sum()
class_weights2 = dict(class_weights2)
class_weights2 = {label_map_old_to_new[k]: v for k, v in class_weights2.items()}

#--- Using build in function
from sklearn.utils.class_weight import compute_class_weight

class_labels = np.unique(fullDS['Class'])
class_weights3 = compute_class_weight(class_weight='balanced', classes=class_labels, y=fullDS['Class'])
class_weights3 = dict(zip(class_labels, class_weights3))
class_weights3 = {label_map_old_to_new[k]: v for k, v in class_weights3.items()}

# Convert all mapped class weights to tensor format
class_weights_tensor1 = torch.tensor([class_weights1[i] for i in sorted(class_weights1.keys())], dtype=torch.float).to(device)
class_weights_tensor2 = torch.tensor([class_weights2[i] for i in sorted(class_weights2.keys())], dtype=torch.float).to(device)
class_weights_tensor3 = torch.tensor([class_weights3[i] for i in sorted(class_weights3.keys())], dtype=torch.float).to(device)

del(class_labels,u_val,counts,val_counts_test,val_counts_train)
#%% load functions

from tqdm import tqdm # to have a progression

#--- Implement early stopping function 
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, validation_loss):
        #assign the first loss
        if self.best_loss is None:
            self.best_loss = validation_loss
        
        #check is the loss is better (lower)
        elif validation_loss < self.best_loss - self.min_delta:
            self.counter = 0
            self.best_loss = validation_loss
            if self.counter >= self.patience:
                self.early_stop = True
        
        # the loss is worse so the add 1 to the counter
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop
    
class EarlyStopper_acc:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_acc = - float('inf')
        self.early_stop = False

    def __call__(self, validation_acc):
        #check is the loss is better (greater)
        if validation_acc >= self.best_acc + self.min_delta:
            self.counter = 0
            self.best_acc = validation_acc
            if self.counter >= self.patience:
                self.early_stop = True
        
        # the loss is worse so the add 1 to the counter
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop
    



def balanced_classification_rate(y_true, y_pred):
    """
    Calculate the Balanced Classification Rate (BCR) in PyTorch.

    Parameters:
    y_true (torch.Tensor): True labels, one-hot encoded (shape: [batch_size, num_classes]).
    y_pred (torch.Tensor): Predicted probabilities or logits (shape: [batch_size, num_classes]).

    Returns:
    float: The BCR value.
    """
    if y_true is None or y_true.shape[0] == 0:
        print(y_true)
        return 0.0

    # Convert predicted logits/probabilities to class predictions
    y_pred_classes = torch.argmax(y_pred, dim=-1)
    y_true_classes = torch.argmax(y_true, dim=-1)
    
    # Calculate correct predictions
    correct_preds = (y_pred_classes == y_true_classes).float()
    
    # Sum of true labels per class
    sum_per_class = y_true.sum(dim=0).float()  # Ensure float type for division
    
    # Calculate accuracy per class
    if len(sum_per_class) == 0:
        return 0.0  # Avoid division by zero if sum_per_class is empty
    
    acc_per_class = torch.zeros_like(sum_per_class, dtype=torch.float)
    for i in range(len(sum_per_class)):
        class_mask = y_true_classes == i
        acc_per_class[i] = correct_preds[class_mask].sum() / max(sum_per_class[i].item(), 1)  # Avoid division by zero
    
    # Compute BCR
    BCR = acc_per_class.mean().item()
    
    return BCR

def per_class_accuracy(y_true, y_pred):
    """
    Calculate the accuracy for each class.

    Parameters:
    y_true (torch.Tensor): True labels, one-hot encoded (shape: [batch_size, num_classes]).
    y_pred (torch.Tensor): Predicted probabilities or logits (shape: [batch_size, num_classes]).

    Returns:
    dict: A dictionary with class indices as keys and corresponding accuracies as values.
    """
    y_pred_classes = torch.argmax(y_pred, dim=-1)
    y_true_classes = torch.argmax(y_true, dim=-1)
    
    unique_classes = torch.unique(y_true_classes)
    class_accuracies = {}
    
    for cls in unique_classes:
        cls_mask = (y_true_classes == cls)
        cls_correct = (y_pred_classes[cls_mask] == y_true_classes[cls_mask]).float().mean().item()
        class_accuracies[int(cls.item())] = cls_correct
    
    return class_accuracies

# Example usage
y_true = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])  # Example true labels (one-hot encoded)
y_pred = torch.tensor([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6], [0.3, 0.5, 0.1]])  # Example predictions

#print(BCR(y_true, y_pred))  #


#%% Train the BERT model  corrected version 



epochs = 100
criterion = torch.nn.CrossEntropyLoss(
    weight= class_weights_tensor1
    )
optimizer = torch.optim.Adam(model_pt.parameters())
early_stopper = EarlyStopper_acc(patience=8)

history = {
    "epoch": [],
    "train_loss": [],
    "valid_loss": [],
    "train_accuracy": [],
    "valid_accuracy": [],
    "train_bcr": [],
    "valid_bcr": []
}

best_valid_acc = 0
best_model_wts = None
best_idx = None

for e in range(epochs):
    start = time()
    model_pt.train() # set the model to training mode
    
    # instantiate variables
    train_loss = 0.0
    train_acc = []
    train_true = []
    train_pred = []
    
    for X, y in tqdm(train_loader):
        X, y = X.to(device), y.to(device)
        prediction = model_pt(X)
        loss = criterion(prediction, y.flatten())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        prediction_index = prediction.argmax(axis=1)
        accuracy = (prediction_index == y.flatten())
        train_acc.extend(accuracy.tolist())  # Use extend to add all items to the list
        
        train_true.append(y)
        train_pred.append(prediction)
    
    train_acc = (sum(train_acc) / len(train_acc))  # Calculate average accuracy for the epoch
    
    # Concatenate all true and predicted values for BCR calculation
    train_true = torch.cat(train_true, dim=0)
    train_pred = torch.cat(train_pred, dim=0)
    train_bcr = balanced_classification_rate(train_true, train_pred) # ??? 2
    
    # Validation phase
    model_pt.eval() # set the model to evaluation mode
    # instantiate variables
    valid_loss = 0.0
    valid_acc = []
    valid_true = []
    valid_pred = []
    
    with torch.no_grad(): # specify no learning is done 
        for X, y in test_loader:
            X, y = X.to(device), y.to(device) # cuda or cpu
            prediction = model_pt(X)
            loss = criterion(prediction, y.flatten())
            
            valid_loss += loss.item()
            
            prediction_index = prediction.argmax(axis=1)
            accuracy = (prediction_index == y.flatten())
            valid_acc.extend(accuracy.tolist())
            
            valid_true.append(y)
            valid_pred.append(prediction)
    
    valid_acc = (sum(valid_acc) / len(valid_acc))  # Calculate average accuracy for the epoch
    train_loss /= len(train_loader)
    valid_loss /= len(test_loader)
    
    # Concatenate all true and predicted values for BCR calculation
    valid_true = torch.cat(valid_true, dim=0)
    valid_pred = torch.cat(valid_pred, dim=0)
    valid_bcr = balanced_classification_rate(valid_true, valid_pred) # ???
    per_class_acc = per_class_accuracy(valid_true, valid_pred) # ??? 2

    history["epoch"].append(e+1)
    history["train_loss"].append(train_loss)
    history["valid_loss"].append(valid_loss)
    history["train_accuracy"].append(train_acc)
    history["valid_accuracy"].append(valid_acc)
    history["train_bcr"].append(train_bcr)
    history["valid_bcr"].append(valid_bcr)
    
    print(f'Epoch {e + 1} \n \t\t Training loss : {train_loss :10.3f} \t\t Validation loss : {valid_loss :10.3f}')
    print(f'\t\t Training Accuracy: {train_acc :10.3%} \t\t Validation Accuracy: {valid_acc :10.3%}')
    print(f'\t\t Training BCR: {train_bcr :10.3f} \t\t Validation BCR: {valid_bcr :10.3f}')
    print(f' \t \t Per-class accuracy: {per_class_acc}')
    
    
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        best_model_wts = model_pt.state_dict().copy()
        best_idx = e + 1 
    
    if early_stopper(valid_acc):
        print(f"Early stopping at epoch {e + 1}")
        break
    
    end = time()
    training_time = round((end - start), 2)
    print(f' \t \t Training time of epoch {e + 1} : {training_time} sec')
    print("*" * 80)


print(f'The best epoch was {best_idx}')

#%% save the best model
model_save_path = "C:/Users/User/Documents/Data Science/LSTAT2380 Statistical Consulting/Project_SWDE/Models_weights/Model_stem_6559.pth"
torch.save({
    'epoch': best_idx,
    'model_state_dict': best_model_wts,
    'best_valid_acc': best_valid_acc,
    'optimizer_state_dict': optimizer.state_dict(),
}, model_save_path)
    
#%% Load the best model weights
model_path = "C:/Users/User/Documents/Data Science/LSTAT2380 Statistical Consulting/Project_SWDE/Models_weights/Model_stem_6559.pth"
checkpoint = torch.load(model_save_path)

# Extract the information
epoch = checkpoint['epoch']
best_valid_acc = checkpoint['best_valid_acc']
model_pt.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

#%% Predictions
from sklearn.metrics import accuracy_score
# model evaluation mode
model_pt.eval()

all_preds = []
all_labels = []

with torch.no_grad():  # Disable gradient computation for prediction
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)  # Move data to the appropriate device (CPU or GPU)
        outputs = model_pt(X)
        _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

# Convert predictions and labels to numpy arrays if needed
all_preds = np.array(all_preds) # predictions
all_labels = np.array(all_labels) # true labels

def get_BCR(y_pred, y_real, num_classes=8):
    # Convert y_real to one-hot encoded format
    y_real_one_hot = np.zeros((y_real.size, num_classes))
    y_real_one_hot[np.arange(y_real.size), y_real] = 1
    
    # Calculate the sum of each class in y_real
    class_sums = np.sum(y_real_one_hot, axis=0)
    
    # Initialize array to store correct predictions for each class
    correct_preds = np.zeros(num_classes)
    
    # Calculate the number of correct predictions for each class
    for i in range(len(y_pred)):
        if y_pred[i] == y_real[i]:
            correct_preds[y_pred[i]] += 1
    
    # Calculate accuracy for each class
    class_accs = correct_preds / class_sums
    
    # Compute the Balanced Classification Rate (BCR)
    BCR = np.mean(class_accs)
    
    # Print individual class accuracies
    for i in range(num_classes):
        print(f"Accuracy for class {i}: {class_accs[i]}")
    
    return BCR

bcr = get_BCR(all_preds, all_labels, num_classes=8)
print(f"Balanced Classification Rate (BCR): {bcr}")

# Calculate accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f'Accuracy: {accuracy:.4f}')