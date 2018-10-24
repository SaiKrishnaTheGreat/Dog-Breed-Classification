# Dog-Breed-Classification
Dog Breed Classification using tensorflow for Kaggle Competetion https://www.kaggle.com/c/dog-breed-identification/kernels

# Transfer-Learning 
The following chart shows how the data flows when using the Inception model for Transfer Learning. First we input and process an image with the Inception model. Just prior to the final classification layer of the Inception model, we save the so-called Transfer Values to a cache-file.The transfer-values are also sometimes called bottleneck-values.

When all the images in the new data-set have been processed through the Inception model and the resulting transfer-values saved to a cache file, then we can use those transfer-values as the input to another neural network. We will then train the second neural network using the classes from the new data-set, so the network learns how to classify images based on the transfer-values from the Inception model.

In this way, the Inception model is used to extract useful information from the images and another neural network is then used for the actual classification.

![alt text](https://github.com/SaiKrishnaTheGreat/Dog-Breed-Classification/blob/master/img/transferLearning.png)

# Implementation

	* Download Files
		Download dataset from https://www.kaggle.com/c/dog-breed-identification/data
		* train.zip
		* test.zip
		* labels.csv.zip
		* sample_submission.zip

	* Setup and Installation
		* Run " sudo pip install -r requirements.txt"
		* Unzip all folders 

	* Data pre_processing
		| python data_processing.py ./
		* Total Images : 10,222
		* Training Images : 9188
		* Validation Images : 1034

	* Train model
		| python retrain.py --image_dir=dataset/ --bottleneck_dir=bottleneck/ --how_many_training_steps=500 --output_graph=trained_model/retrained_graph.pb --output_labels=trained_model/retrained_labels.txt --summaries_dir=summaries --print_misclassified_test_images

	* Test model
		| python identiy_dog.py <input_image>

	* Generate Test Report 
		| python generate_dog_breed_report.py

# Results  
	* Epochs = 2500
	* Training Time = 12.57min on i5.
	* Accuracy = 91.0%  out of 1034 images 941 are identified correctly.
![alt text](https://github.com/SaiKrishnaTheGreat/Dog-Breed-Classification/blob/master/img/result_1.png)

# How to Improve the accuracy
	* Adding more Data
		Having more data is always a good idea. It allows the “data to tell for itself,” instead of relying on assumptions and weak correlations. Presence of more data results in better and accurate models
	* Model selection 
		Other models like ResNet and MobileNet can give more accurate results.
	* Data Filtration
		Increasing the existing dataset with rotation,color filtration can insrease accuracy.
	* Algorithm selection OR Traing from scratch
		Instead of Transferlearning, training from scratch(though takes more time) can increase the accuracy as the calculated parameters decreases.
