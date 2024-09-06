# Deep-Learning-Projects
My solutions to IUST's Deep Learning Mini Projects, Fall 2023, Dr. Davoodabadi.

## <img width="40" height="40" src="https://img.icons8.com/?size=100&id=kOPTH4LnJoIU&format=png&color=000000" alt="homework"/> Projects
### P1
- <img width="40" height="40" src="https://img.icons8.com/?size=100&id=104091&format=png&color=000000" /> Machine Learning and Deep Learning Project Solutions
This repository contains solutions to various machine learning and deep learning problems, including:

•  Probabilistic Models: Calculating class probabilities and classifying news headlines using learned probabilistic models.

•  Laplace Smoothing: Handling new data in test news using Laplace smoothing with an alpha coefficient of 1.

•  Logistic Regression: Modeling conditional distributions and applying maximum likelihood estimation.

•  Activation Functions: Explaining and comparing activation functions like Sigmoid, Softmax, ReLU, and Tanh.

•  MLP Design: Designing and implementing a simple MLP network to classify images, including architecture selection and justification.

•  Two-Class Classification: Using a multi-layer perceptron (MLP) for two-class classification and discussing potential challenges.

•  Machine Learning vs. Deep Learning: Analyzing key differences and discussing the effectiveness of deep vs. wide networks.

- Answers: [Link to DL1](https://github.com/Sarayounesi/Deep-Learning-Projects/tree/main/DL_1_SaraYounesi)

### P2
- 
----------------------------------------------------------------------------------------------------
Overfitting and Underfitting in Neural Networks
•  Explanation of overfitting and underfitting.

•  Methods to detect overfitting in pre-trained models.

•  Application of Dropout to prevent overfitting, with practical calculations.

----------------------------------------------------------------------------------------------------
K-Nearest Neighbors (KNN) Algorithm
•  Impact of changing the value of K on bias and variance.

•  Analysis of regularization effects on model performance.

•  Identification of L1 and L2 regularization through experimental results.

1. 
Knowledge Distillation
•  Explanation of the knowledge distillation process.

•  Learning process from teacher to student network.

•  Analysis of loss functions for updating student network weights.

1. 
Optimization and Model Training
•  Comparison of different optimizers using provided code.

•  Implementation and training of a simple MLP for FashionMNIST dataset.

•  Techniques to induce and mitigate overfitting, including data augmentation and regularization.
- Answers: [Link to HW2](https://github.com/Sarayounesi/Deep-Learning-Projects/tree/main/DL_2_SaraYounesi)

### P3
- Optimization and Convolutional Neural Networks Solutions
This repository contains solutions to various optimization and convolutional neural network problems, including:

1. 
Optimization Techniques
•  Issues with high and low learning rates and their detection.

•  Comparison of Adam and SGD algorithms in handling saddle points.

•  Analysis of cost reduction using different optimization algorithms.

1. 
Convolutional Layers and Backpropagation
•  Calculation of gradients for convolutional layers using backpropagation.

•  Parameter count for different network layers.

•  Comparison and applications of 2D and 3D convolutional layers.

1. 
Brain Tumor Classification
•  Implementation of a CNN for brain tumor classification.

•  Use of Sequential and Functional API architectures.

1. 
Practical Applications of Convolutional Layers
•  Real-world examples where convolutional layers excel.

•  Challenges posed by convolutional layers in certain scenarios.

1. 
1x1 Convolutional Filters
•  Purpose and impact of 1x1 filters in CNNs.

•  Implementation and analysis of a simple CNN with 1x1 filters.

1. 
Inception Module Implementation
•  Structure and purpose of the Inception module.

•  Impact of stride parameter on spatial dimensions.

•  Key features and operations of convolutional layers in the network.
- Answers: [Link to HW3](https://github.com/Sarayounesi/Deep-Learning-Projects/tree/main/DL_3_SaraYounesi)

### P4
- MNIST Classification and Neural Network Optimization
This repository contains solutions to various machine learning and neural network problems, including:

1. 
MNIST Dataset Classification
•  Design and optimization of a convolutional neural network (CNN) using Keras Tuner.

•  Explanation of Keras Tuner and its application for hyperparameter optimization.

•  Implementation of dropout and pooling layers to improve network performance.

1. 
Medical Data Analysis
•  Completion of tasks in the provided ipynb.medical notebook for medical data analysis.
2. 
Gas Consumption Prediction
•  Analysis and correction of an LSTM model for predicting gas consumption.

•  Identification of issues causing negative R² and suggestions for improvement.

1. 
Convolutional and Recurrent Neural Networks
•  Comparison of CNNs and RNNs, their applications, and parameter counts.

•  Parallelization capabilities of CNNs vs. RNNs.

1. 
CNN Layer Analysis
•  Calculation of output sizes and parameter counts for various CNN layers.

•  Determination of padding required for dilated convolutions.

1. 
Batch Normalization
•  Explanation of batch normalization and its impact on training.

•  Completion of batch normalization code using NumPy.

•  Analysis of the role of the epsilon hyperparameter.

1. 
Grad-CAM Implementation
•  Implementation of Grad-CAM algorithm on the MNIST dataset.

•  Visualization and interpretation of Grad-CAM results.
- Answers: [Link to HW4](https://github.com/Sarayounesi/Deep-Learning-Projects/tree/main/DL_4_SaraYounesi)

### P5
RNNs, Backpropagation, and Attention Mechanisms
This repository contains solutions to various machine learning and neural network problems, including:

1. 
RNN Architectures and Applications
•  Suitable tasks for one-to-many RNN architectures.

•  Comparison of unidirectional and bidirectional RNNs for different use cases.

•  Training RNN language models and their predictions.

1. 
Backpropagation in RNNs
•  Calculation of gradients for RNNs using backpropagation.

•  Detailed derivation of gradients with respect to weights and activations.

1. 
Attention Mechanisms
•  Implementation of a hypothetical "argmax" attention mechanism.

•  Analysis of the impact of using argmax on model training and gradient flow.

1. 
Practical Exercises
•  Completion of tasks in the provided ipynb4.Question notebook.

- Answers: [Link to HW5](https://github.com/Sarayounesi/Deep-Learning-Projects/tree/main/DL_5_SaraYounesi)

### P6
Convolutional and Attention-based Networks, Evaluation Metrics, and GANs
This repository contains solutions to various machine learning and neural network problems, including:

1. 
Convolutional vs. Attention-based Networks
•  Performance comparison for tasks like cat vs. non-cat and human vs. non-human classification.

•  Brief explanations of how each network type handles these tasks.

1. 
Evaluation Metrics
•  Explanation of TP, FP, TN, and FN concepts.

•  Recommended evaluation metrics for projects with high stakes, such as criminal detection.

1. 
Rotation Estimation and One-Hot Vectors
•  Benefits of rotation estimation for classification tasks.

•  Explanation of one-hot vectors and their limitations.

•  Discussion on how Word2Vec aligns with self-supervised learning algorithms.

1. 
Network Structure and Hyperparameter Search
•  Overview of reinforcement learning for automatic network and hyperparameter search.

•  Applicability of this approach for object detection tasks, including input image size and layer count.

1. 
GAN Training
•  Analysis of why generated image quality may differ between the first and 100th epoch, despite similar loss values.

- Answers: [Link to HW6](https://github.com/Sarayounesi/Deep-Learning-Projects/tree/main/DL_6_SaraYounesi)



## <img width="20" height="20" src="https://img.icons8.com/external-smashingstocks-mixed-smashing-stocks/68/41b883/external-Notes-work-from-home-smashingstocks-mixed-smashing-stocks-2.png" alt="Notes"/> Notes
- Description: Lecture slides provided by the professor.
- [Link to Notes](https://github.com/lelnazrezaeel/Computer-Vision-IUST/tree/main/Notes)
