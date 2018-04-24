''' Convolutional Autoencoder '''
import os
import numpy as np
import logging
import h5py
import argparse
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from torchvision.utils import save_image

''' Define parser'''
parser = argparse.ArgumentParser()
parser.add_argument('-c', help="class", dest="c")
args = parser.parse_args()
print(args.c)

''' Parameters '''
class_model = args.c
file_dataset = 'images.h5'

''' Hyperparameters '''
batch_size = 1
num_epochs = 100
learning_rate = 1e-3

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

def to_img(x):
    x = x.view(x.size(0), 1, 220, 220)
    return x


''' Define paths '''
path = dict()
path['dataset'] = 'C:\datasets\FreeFormDeformation'
path['model'] = 'C:\data\img2mesh'
filename_dataset = os.path.join(path['dataset'], class2uid[class_model], 'rendered', 'parameters',
                                file_dataset)
filename_model = os.path.join(path['model'], class2uid[class_model], 'autoencoder_images')
filename_log = os.path.join(path['model'], class2uid[class_model],
                            'model_cae_b'+str(batch_size)+'_e'+str(num_epochs)+'.log')
if not os.path.exists(filename_model):
    os.makedirs(filename_model)


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


''' Manual seed for reproducibility '''
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(666) # GPU seed
else:
    torch.manual_seed(666) # CPU seed


''' Load dataset '''
# Load train + labels (GT) from a h5 file
hf = h5py.File(filename_dataset, 'r')
dataset = np.array(hf.get('IN'), dtype=np.float32).T / 255. # image normalized to 0 - 1
hf.close()

# Crop images to have a square aspect ratio
imgs_dataset = []
x_offset = 22
y_offset = 54
for i in range(len(dataset)):
    img = dataset[i].reshape((256,192)) # vector to image size
    img_aux = np.full((300, 300), 1.)
    img_aux[x_offset:img.shape[0]+x_offset, y_offset:img.shape[1]+y_offset] = img
    img_aux = img_aux[40:-40, 40:-40]
    imgs_dataset.append(img_aux.T)

imgs_dataset = np.expand_dims(np.array(imgs_dataset), 1)

train_dataset = np.array(imgs_dataset[:3500,:,:,:], dtype=np.float32) # 70%
test_dataset = np.array(imgs_dataset[3500:,:,:,:], dtype=np.float32) # 30%

# Transform to torch tensors
train_dataset = torch.from_numpy(train_dataset)
test_dataset = torch.from_numpy(test_dataset)


''' Make dataset iterable '''
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = False)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = batch_size,
                                          shuffle = False)


''' Create model class '''
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=3, padding=0), # 8,72,72
            nn.ReLU(True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=3, padding=0), # 16,24,24
            nn.ReLU(True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=3, padding=0), # 8,8,8
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=3, padding=0), # 16,24,24
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=5, stride=3, padding=0), # 1,72,72
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=5, stride=3, padding=2),  # 1,220,220
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


''' Instantiate model class '''
model = ConvAutoencoder()
print(model)
#  Use GPU for model
if torch.cuda.is_available():
    model.cuda()

''' Instantiate loss class '''
criterion = nn.MSELoss()

''' Instantiate optimizer class '''
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)


''' Train the model '''
logging.info('Train')
print('Training the model...')
for epoch in range(num_epochs):
    losses = []
    for train_dataset in train_loader:
        #  Use GPU for model
        if torch.cuda.is_available():
            inputs = Variable(train_dataset).cuda()
        else:
            inputs = Variable(train_dataset)

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
        # Forward pass to get output/logits
        outputs = model(inputs)
        # Calculate loss
        loss = criterion(outputs, inputs)
        # Getting gradients w.r.t. parameters
        loss.backward()
        # Updating parameters
        optimizer.step()

        losses.append(loss.data.mean())

    print('Epoch: [%d/%d], Loss: %.5f' % (epoch+1, num_epochs, np.mean(losses)))
    logging.info('Epoch: [%d/%d], Loss: %.5f', epoch+1, num_epochs, np.mean(losses))

print('Finished training')

# save some images
save_image(to_img(outputs[0:5,0,:,:].data.cpu()), os.path.join(filename_model, 'image_train.png'))
save_image(to_img(inputs[0:5,0,:,:].data.cpu()), os.path.join(filename_model, 'image_train_gt.png'))


''' Test the model '''
logging.info('Test')
model.eval()
losses = []
print('Testing the model...')
for test_dataset in test_loader:
    #  Use GPU for model
    if torch.cuda.is_available():
        inputs = Variable(test_dataset.cuda())
    else:
        inputs = Variable(test_dataset)

    outputs = model(inputs)
    loss = criterion(outputs, inputs)
    losses.append(loss.data.mean())

print('Loss: %.5f' % (np.mean(losses)))
logging.info('Loss: %.5f', np.mean(losses))

print('Finished testing')


''' Save the Model '''
print('Saving the model...')
filename = os.path.join(path['model'], class2uid[class_model],
                        'model_cae_b'+str(batch_size)+'_e'+str(num_epochs)+'.pkl')
torch.save(model.state_dict(), filename)
print('Model saved')


''' Get descriptors from the encoder '''
print('Getting descriptors...')
# Load trained weights
# model.load_state_dict(torch.load(filename))

model = model.encoder
print(model)
model.eval()

# Get train descriptors and save it
train_descriptors = []
for train_dataset in train_loader:
    #  Use GPU for model
    if torch.cuda.is_available():
        inputs = Variable(train_dataset).cuda()
    else:
        inputs = Variable(train_dataset)

    outputs = model.forward(inputs)
    train_descriptors.append(outputs.data.cpu().numpy().flatten())

# print(outputs.data.cpu().numpy().shape)
# print(np.array(train_descriptors).shape)

filename = os.path.join(path['model'], class2uid[class_model],
                        'model_cae_descriptors_b'+str(batch_size)+'_e'+str(num_epochs)+'_train.h5')
with h5py.File(filename, 'w') as hf:
    hf.create_dataset("descriptors",  data=train_descriptors)

# Get test descriptors and save it
test_descriptors = []
for test_dataset in test_loader:
    #  Use GPU for model
    if torch.cuda.is_available():
        inputs = Variable(test_dataset).cuda()
    else:
        inputs = Variable(test_dataset)

    outputs = model.forward(inputs)
    test_descriptors.append(outputs.data.cpu().numpy().flatten())

# print(outputs.data.cpu().numpy().shape)
# print(np.array(test_descriptors).shape)

filename = os.path.join(path['model'], class2uid[class_model],
                        'model_cae_descriptors_b'+str(batch_size)+'_e'+str(num_epochs)+'_test.h5')
with h5py.File(filename, 'w') as hf:
    hf.create_dataset("descriptors",  data=test_descriptors)

print('Finally done!')
