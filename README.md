# Sagittal Classificaiton

## Approach

We approached the problem in two general directions, a deep learning varient and a more traditional computer vision + machine learning varient.

### Deep Learning

The [Deep Learning](CNN/) approach used a convolutional neural network trained on hand-labeled data to determine if the image was of sagittal planar orient. The image was cropped to only show the relevant information. The image was then resized to a standard format and fed into the CNN. The CNN architecture was varited but the total accuracy was ~93%. 

The model was trained on a NVIDIA GTX 1080 using Keras. The model was trained for at most 100 epochs using patience set on the validation loss.  

### Machine Learning

The [Machine Learning](ML/) approach found features using the HOG (Histogram of Oriented Gradients) approach. The images were originally cropped, scaled down, and the HOG transform was applied. Then a variety of ML models were trained on the new labeled information. First, individual SVM, Random Forests, Linear Regression, etc was trained, and an ensemble network was also trained. The total accuracy was ~89%. 

## Future Steps

We were looking towards using Binary Thresholding as another base for a method of classification. 
