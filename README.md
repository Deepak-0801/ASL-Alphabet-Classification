ASL Alphabet Classifier
=======================

This program trains a convolutional neural network (CNN) using TensorFlow to classify images of American Sign Language (ASL) alphabet letters. The dataset used to train the model contains images of each letter in the ASL alphabet.

Requirements
------------

-   Python 3.6 or later
-   TensorFlow 2.x
-   Keras

Usage
-----

1.  Clone this repository
2.  Download the ASL Alphabet dataset and extract it into a directory called `asl_alphabet_dataset` in the same directory as this README file.
3.  Run the program using `python asl_alphabet_classifier.py`.

Program Details
---------------

The program loads the ASL Alphabet dataset using the `ImageDataGenerator` class from Keras. The images are scaled, sheared, zoomed, and flipped horizontally for data augmentation. The dataset is split into training and validation sets using a 75/25 split.

The model used is a CNN with two convolutional layers, two max pooling layers, and two dense layers. The model is compiled with the Adam optimizer and categorical crossentropy loss. The program trains the model for 10 epochs and evaluates the model on a separate test set.

Results
-------

After training the model, the program evaluates it on the test set and prints out the loss and accuracy scores. The results of the program may vary depending on the hardware and the number of epochs used for training.

Improvements
------------

There are several ways to improve the performance of the ASL alphabet classifier. Here are a few suggestions:

-   Increase the number of epochs for training the model.
-   Experiment with different architectures for the model.
-   Try different combinations of hyperparameters such as batch size and learning rate.
-   Use transfer learning to fine-tune a pre-trained model on a larger dataset.

Conclusion
----------

In conclusion, this program demonstrates how to use TensorFlow and Keras to build a CNN for classifying images of the ASL alphabet letters. The program can be further improved by tweaking the hyperparameters, using different architectures, and fine-tuning a pre-trained model.

Credits
-------

The ASL Alphabet dataset used in this program was created by [tejaswinpotnuru](https://github.com/tejaswinpotnuru) and can be found on [Kaggle](https://www.kaggle.com/datamunge/sign-language-mnist). The code for this program was written by [Deepak](https://github.com/Deepak-0801).