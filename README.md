# Dog-Breed-Classification
Dog Breed Classification using tensorflow for Kaggle Competetion https://www.kaggle.com/c/dog-breed-identification/kernels

# Transfer-Learning 
The following chart shows how the data flows when using the Inception model for Transfer Learning. First we input and process an image with the Inception model. Just prior to the final classification layer of the Inception model, we save the so-called Transfer Values to a cache-file.The transfer-values are also sometimes called bottleneck-values.

When all the images in the new data-set have been processed through the Inception model and the resulting transfer-values saved to a cache file, then we can use those transfer-values as the input to another neural network. We will then train the second neural network using the classes from the new data-set, so the network learns how to classify images based on the transfer-values from the Inception model.

In this way, the Inception model is used to extract useful information from the images and another neural network is then used for the actual classification.

![alt text](https://github.com/SaiKrishnaTheGreat/Dog-Breed-Classification/blob/master/img/transferLearning.png)

