''' Feedforward Net to Learn the Graph Parameters (FFD, LC) '''
import os
import numpy as np
import logging
import h5py
import argparse
import scipy.io
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable

''' Define parser'''
parser = argparse.ArgumentParser()
parser.add_argument('-c', help="class", dest="c")
args = parser.parse_args()
print(args.c)


''' Parameters '''
class_model = args.c
file_labels = 'FFD_LC_IDX_params.h5'
file_descriptor_train = 'model_cae_descriptors_b1_e100_train.h5'
file_descriptor_test = 'model_cae_descriptors_b1_e100_test.h5'

''' Hyperparameters '''
batch_size = 1
num_epochs = 2000
input_dim = 2048 # 8*8 * 32 kernels
hidden_dim = 1500 # ~(input_dim + output_dim) / 2
output_dim = 126
learning_rate = 0.001


''' Define some tools '''
# Dict to convert class to UID (Shapenet)
class2uid = {
    'bottle'        : '02876657',
    'bicycle'       : '02834778',
    'knife'         : '03624134',
    'chair'         : '03001627',
    'car'           : '02958343',
    'diningtable'   : '04379243',
    'sofa'          : '04256520',
    'bed'           : '02818832',
    'dresser'       : '02933112',
    'aeroplane'     : '02691156',
    'motorbike'     : '03790512',
    'bus'           : '02924116',
}

''' Define paths '''
path = dict()
path['dataset'] = 'C:\datasets\FreeFormDeformation'
path['model'] = 'C:\data\img2mesh'
filename_train = os.path.join(path['model'], class2uid[class_model], file_descriptor_train)
filename_test = os.path.join(path['model'], class2uid[class_model], file_descriptor_test)
filename_labels = os.path.join(path['dataset'], class2uid[class_model], 'rendered', 'parameters',
                            file_labels)
filename_log = os.path.join(path['model'], class2uid[class_model],
                            'model_ffn_b'+str(batch_size)+'_e'+str(num_epochs)+'.log')


''' Logger setup '''
logging.basicConfig(
    filename = filename_log,
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    filemode = 'w'
)
logging.getLogger(__name__)
logging.info('Model Parameters')
logging.info('Class model : %s ' % (class_model))
logging.info('Batch size : %d ' % (batch_size))
logging.info('Num Epochs : %d ' % (num_epochs))
logging.info('Learning Rate : %f ' % (learning_rate))
logging.info('Input dimension : %f ' % (input_dim))
logging.info('Output dimension : %f ' % (output_dim))


''' Manual seed for reproducibility '''
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(666) # GPU seed
else:
    torch.manual_seed(666) # CPU seed


''' Load dataset '''
# Load train from a h5 file
hf = h5py.File(filename_train, 'r')
train_dataset = np.array(hf.get('descriptors'), dtype=np.float32)
hf.close()

# Load test from a h5 file
hf = h5py.File(filename_test, 'r')
test_dataset = np.array(hf.get('descriptors'), dtype=np.float32)
hf.close()

# load labels
hf = h5py.File(filename_labels, 'r')
dataset_labels = hf.get('GT')
train_dataset_labels = np.array(dataset_labels[0:126,:3500], dtype=np.float32).T
test_dataset_labels = np.array(dataset_labels[0:126,3500:], dtype=np.float32).T
hf.close()

# Transform to torch tensors
train_dataset = torch.from_numpy(train_dataset)
test_dataset = torch.from_numpy(test_dataset)
train_dataset_labels = torch.from_numpy(train_dataset_labels)
test_dataset_labels = torch.from_numpy(test_dataset_labels)

''' Put data + labels into tuples '''
train_dataset_tuple = np.empty(3500, dtype=list)
for i in range(0, 3500):
    train_dataset_tuple[i] = (train_dataset[i], train_dataset_labels[i])

test_dataset_tuple = np.empty(1500, dtype=list)
for i in range(0, 1500):
    test_dataset_tuple[i] = (test_dataset[i], test_dataset_labels[i])


''' Making dataset iterable '''
train_loader = torch.utils.data.DataLoader(dataset = train_dataset_tuple,
                                           batch_size = batch_size,
                                           shuffle = False)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset_tuple,
                                          batch_size = batch_size,
                                          shuffle = False)


''' Create model class '''
class FeedforwardNeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


''' Instantiate model class '''
model = FeedforwardNeuralNet(input_dim, hidden_dim, output_dim)
print(model)
#  Use GPU for model
if torch.cuda.is_available():
    model.cuda()

''' Instantiate loss class '''
criterion = nn.MSELoss()

'''
Instantiate optimizer class '''
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


''' Train the model '''
logging.info('Train')
print('Training the model...')
for epoch in range(num_epochs):
    losses = []
    for i, (train_dataset, train_dataset_labels) in enumerate(train_loader):
        #  Use GPU for model
        if torch.cuda.is_available():
            inputs = Variable(train_dataset.cuda())
            labels = Variable(train_dataset_labels.cuda())
        else:
            inputs = Variable(train_dataset)
            labels = Variable(train_dataset_labels)

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
        # Forward pass to get output/logits
        outputs = model(inputs)
        # Calculate loss
        loss = criterion(outputs, labels)
        # Getting gradients w.r.t. parameters
        loss.backward()
        # Updating parameters
        optimizer.step()

        losses.append(loss.data.mean())

    print('Epoch: [%d/%d], Loss: %.5f' % (epoch+1, num_epochs, np.mean(losses)))
    logging.info('Epoch: [%d/%d], Loss: %.5f', epoch+1, num_epochs, np.mean(losses))

print('Finished training')

print(outputs)
print(labels)
print(np.linalg.norm(outputs-labels))


''' Test model '''
logging.info('Test')
model.eval()
losses = []
print('Testing the model...')
for i, (test_dataset, test_dataset_labels) in enumerate(test_loader):
    #  Use GPU for model
    if torch.cuda.is_available():
        inputs = Variable(test_dataset.cuda())
        labels = Variable(test_dataset_labels.cuda())
    else:
        inputs = Variable(test_dataset)
        labels = Variable(test_dataset_labels)

    outputs = model(inputs)
    loss = criterion(outputs, labels)
    losses.append(loss.data.mean())

print('Loss: %.5f' % (np.mean(losses)))
logging.info('Loss: %.5f', np.mean(losses))

print('Finished testing')


''' Save the Model '''
print('Saving the model...')
filename = os.path.join(path['model'], class2uid[class_model],
                        'model_fnn_b'+str(batch_size)+'_e'+str(num_epochs)+'.pkl')
torch.save(model.state_dict(), filename)
print('Model saved')


''' Get the estimated parameters (FFD + LC) '''
print('Getting the estimated parameters...')
# Load trained weights
# model.load_state_dict(torch.load(filename))

model.eval()

# Get the train final parameters and save it
params_train = []
for i, (train_dataset, train_dataset_labels) in enumerate(train_loader):
    #  Use GPU for model
    if torch.cuda.is_available():
        inputs = Variable(train_dataset.cuda())
        labels = Variable(train_dataset_labels.cuda())
    else:
        inputs = Variable(train_dataset)
        labels = Variable(train_dataset_labels)

    outputs = model(inputs)
    loss = criterion(outputs, labels)
    params_train.append(outputs.data.cpu().numpy())

filename = os.path.join(path['model'], class2uid[class_model],
                        'model_fnn_params_b'+str(batch_size)+'_e'+str(num_epochs)+'_train.mat')
scipy.io.savemat(filename, {'params_ffd_train': params_train})

# Get the test final parameters and save it
params_test = []
for i, (test_dataset, test_dataset_labels) in enumerate(test_loader):
    #  Use GPU for model
    if torch.cuda.is_available():
        inputs = Variable(test_dataset.cuda())
        labels = Variable(test_dataset_labels.cuda())
    else:
        inputs = Variable(test_dataset)
        labels = Variable(test_dataset_labels)

    outputs = model(inputs)
    loss = criterion(outputs, labels)
    params_test.append(outputs.data.cpu().numpy())

filename = os.path.join(path['model'], class2uid[class_model],
                        'model_fnn_params_b'+str(batch_size)+'_e'+str(num_epochs)+'_test.mat')
scipy.io.savemat(filename, {'params_ffd_test': params_test})

print('Finally done!')
