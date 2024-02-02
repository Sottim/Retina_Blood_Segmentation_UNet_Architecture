# Retina_Blood_Segmentation_UNet_Architecture

## Overview
This repository contains my work for Image(Semantic) Segmentation for Retina Blood Vessel. Through this project, I aim to learn various deep learning concepts and algorithms. The architecture in this project is : [U-Net](https://arxiv.org/abs/1505.04597) Architecture which is popularly used for Biomedical Image Segmentation. The architecture consists of a contracting path (Encoders) to capture context and a symmetric expanding path (Decoders) that enables precise localization. 

## Dataset 
The Digital Retinal Images for Vessel Extraction (DRIVE) dataset is a dataset for retinal vessel segmentation. It consists of a total of JPEG 40 color fundus images; including 7 abnormal pathology cases. 
The set of 40 images was equally divided into 20 images for the training set and 20 images for the testing set.
![Image](./new_data/test/image/01_test_0.png)
![Image Mask](./new_data/test/mask/01_test_0.png)

## Coding Approach
### 1. Data Augumentation
Since the dataset was very small (20 images for the training set and 20 images for the testing set), data augumentation was used to augument the training set to increase the training set to 80 images in total using (HorizontalFlip, VerticalFlip, Rotate) for each 20 images. 

### 2. Folder : New Data
It contains the new training set (with augumented data) and the original test set along with each of their binary mask.

### 3. model.py
It implements the architecture of U-Net taking the advantage of convolution blocks, encoder block, Bottleneck (bridge layer) and the decorder block. 

1. Encoder block is basically a convolutional block followed by the pooling     layer i.e maxpooling. Basically, the spatial dimention i.e height and width are reduced as we go from encoder1, encoder2 ...
And Number of filter i.e 64 increases to 128, 256 as we go. 

2. Decorder block : It starts with 2*2 transpose convolutional. Takes input channel, output channel and since we need to upsample (increase) features height and width by stride of 2.  For the first decorder block, it takes first input as the output of the bottleneck and skip conncetion.

### 4. train.py
During the model training, split the augumented dataset in (New Data Folder) into training and validation sets where 15% of 80 = 12 images are used for validation during training. Train the model, evaluate the accuracy and save the model.

### 5. test.py
The model is tested and we jaccard metrics score is used to highlight the model's precision. 

#### Future
Due to the computational limitation, the training was not fully complete. However, upon testing the model shows the promising results within few epochs. In future, I aim to make the model more robust by incroperating the Res-Net Architecture with UNet.

Still more to go...






