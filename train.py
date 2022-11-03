from __future__ import print_function

# Global Imports
import torch

from os.path import join
from torch import nn
from torch.autograd import Variable

# Local Imports
import tb_logger

from data import readSEGY, readLabels, get_slice
from batch import get_random_batch
from texture_net import TextureNet
from utils import computeAccuracy, interpret, var_to_np

# Parameters
dataset_name = 'F3'
device_name = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
image_size = 65
batch_size = 32 #If you have a GPU with little memory, try reducing this to 16 (may degrade results)
log_tensorboard = True #Log progress on tensor board
num_iterations = 100

if log_tensorboard: 
    logger = tb_logger.TBLogger('log', 'Train')

#See the texture_net.py file for the network configuration
network = TextureNet(n_classes=2)

#Loss function
cross_entropy = nn.CrossEntropyLoss() #Softmax function is included

#Optimizer to control step size in gradient descent
optimizer = torch.optim.Adam(network.parameters())

#Transfer model to gpu
network = network.to(device_name)

#Load the data cube and labels
data, data_info = readSEGY(join(dataset_name,'data.segy'))
train_class_imgs, train_coordinates = readLabels(join(dataset_name,'train'), data_info)
val_class_imgs, _ = readLabels(join(dataset_name,'val'), data_info)

#Plot training/validation data with labels
if log_tensorboard:
    for class_img in train_class_imgs + val_class_imgs:
        logger.log_images(class_img[1] + '_' + str(class_img[2] ), get_slice(data, data_info, class_img[1], class_img[2]), cm='gray')
        logger.log_images(class_img[1] + '_' + str(class_img[2]) + '_true_class', class_img[0])


# Training loop
for i in range(2000):

    # Get random training batch with augmentation
    # This is the bottle-neck for training and could be done more efficient on the GPU...
    [batch, labels] = get_random_batch(data, train_coordinates, image_size, batch_size,
                                       random_flip=True,
                                       random_stretch=.2,
                                       random_rot_xy=180,
                                       random_rot_z=15)


    #Format data to torch-variable
    batch = Variable( torch.Tensor(batch).float() )
    labels = Variable( torch.Tensor(labels).long() )

    # Transfer data to gpu
    batch = batch.to(device_name)
    labels = labels.to(device_name)

    #Set network to training phase
    network.train()

    #Run the samples through the network
    output = network(batch)

    #Compute loss
    loss = cross_entropy( torch.squeeze(output) , labels)

    # Do back-propagation to get gradients of weights w.r.t. loss
    loss.backward()

    # Ask the optimizer to adjust the parameters in the direction of lower loss
    optimizer.step()

    # Every 10th iteration - print training loss
    if i % 10 == 0:
        network.eval()

        #Log to training loss/acc
        print('Iteration:', i, 'Training loss:', var_to_np(loss))
        if log_tensorboard:
            logger.log_scalar('training_loss',  var_to_np(loss),i)
        for k,v in computeAccuracy(torch.argmax(output,1), labels).items():
            if log_tensorboard:
                logger.log_scalar('training_' + k, v, i)
            print(' -',k,v,'%')

    #every 100th iteration
    if i % 100 == 0 and log_tensorboard:
        network.eval()

        # Output predicted train/validation class/probability images
        for class_img in train_class_imgs + val_class_imgs:

            slice = class_img[1]
            slice_no = class_img[2]

            class_img = interpret(network.classify, data, data_info, slice, slice_no, image_size, 16, return_full_size=True, device=device_name)
            logger.log_images( slice + '_' + str(slice_no)+ '_pred_class', class_img, i)

            class_img = interpret(network, data, data_info, slice, slice_no, image_size, 16, return_full_size=True, device=device_name)
            logger.log_images( slice + '_' + str(slice_no) + '_pred_prob', class_img, i)

        #Store trained network
        torch.save(network.state_dict(), join(dataset_name, 'saved_model.pt'))




