# AICommons - AI4Good Competition 
## Edge-based early detection and alerting systems for Forest Guards - Team Lisa

<hr>

[Repository Tree](#repo-tree) | [Datasets Used](#datasets-used) | [Datasets Generated](#datasets-generated) | [Notebooks](#notebooks) | [Bottleneck Features](#bottleneck-features) | [Models](#models) | [Mobile App Client](#mobile-app-client) | [Team Members](#team-members) | [References](#references)

<hr>

## Repo Tree

	
	├───apk
		└───forest_guard_client.apk

	├───labels_and_classescsv
		├───ESC-50_Mel-Spectogram_dataset_Meta_data.csv
		├───ESC-50_WavePlot_dataset_Meta_data.csv
		└───labels_and_target_classes(ESC50).csv
	
	├───models
		├───ForestAI.tflite
		└───illegal_logging_classifier_model.h5
	
	├───notebooks
		├───Dataset_exploration_and__Generation_by_Author_Appau_Ernest.ipynb
		├───iteration1_Building_a_CNN_model_to_classify_audio_events_from_the_ESC_50_WavePlot_dataset.ipynb
		└───Iteration2_Building_a_CNN_model_to_classify_audio_events_from_the_ESC_50_MelSpectogram_dataset.ipynb

	├───pickle_files
		├───test_data.pkl
		├───test_labels.pkl
		├───train_labels.pkl
		├───train_data.pkl
		├───validation_labels.pkl
		└───validation_data.pkl
		
	├───forest-guard-ai4good-firebase-adminsdk-1fw9p-232da17716.json
	
	└───Testing_final_model_Team_Lisa.ipynb


> Open [Testing_final_notebook](https://colab.research.google.com/drive/1IuMvh0sCHniIwgHHNCUNXeH3SWuscEp0?usp=sharing) to make a copy of this notebook to run sample test in Colab. It contains an end to end pipeline with instructions to run inference on sound inputs.

## Datasets Used
1. ESC-50 Dataset (Primary): https://github.com/karolpiczak/ESC-50
2. Sampled Audio For Testing (Secondary): https://research.google.com/audioset

## Datasets Generated
1. ESC-50 waveplot/amplitude portfolio dataset
2. ESC-50 MelSpectogram portfolio dataset

## Notebooks

1. **Notebook 1**: [Open In Colab](https://colab.research.google.com/drive/1cUb_4mp9zQujwEs75m7ZVRKIzXieBqsq?usp=sharing)
This notebook involves a feature engineering approach to generate two distinct datasets
to help us in the modelling process in an attempt to solve the challenge of illegal logging activities in the forests.
Generated Datasets include:
	- ESC-50 Waveplot dataset
	- ESC-50 MelSpectogram dataset

2. **Notebook 2**: [Open In Colab](https://colab.research.google.com/drive/1seJqPEI2YLDNASsPrShS85QRSKIxrHvK?usp=sharing)
The first iteration by building a base Convolutional Neural Network to classify 
audio waveplot/amplitude profile.
Dataset used in this notebook includes the Generated ESC-50 waveplot profiles of the original audio files.  
Results from this iteration was a poor performing model with a high bias and variance between training and validation splits. In conclusion, we moved on to the melspectogram dataset to achieve a better performance after dozen hyperparameter tuning exercises.

3. **Notebook 3**: [Open In Colab](https://colab.research.google.com/drive/1lN8Z5dbJYTT8Y531shL3YyFTEe93H1Kf?usp=sharing)
In this notebook, our model performed a bit better than the previous model in notebook 2 but also displayed a high variance during training process.  
After a dozen hyperparameter tuning and regularization, we concluded on the lack of our generated dataset not well feature-engineered enough to accommodate a generalized model.

4. **Notebook 4**: [Open In Colab](https://colab.research.google.com/drive/1p7iOe5fQz6TeK1YUWbf4A5CbkgsSdn_S?usp=sharing)
We adopted an advanced feature engineering approach by converting our audio files to a concatenated stack of 3 channels based on three main operations to generate feature maps.
This includes splitting the 5 second audio clips into 5 folds increasing the dataset from 2000 to 10000 audio files. We combined three feature maps which include the Mel spectogram, log-scaled spectogram and the delta mel-scaled spectogram to form one feature map profile for an audio clip.

5. **Notebook 5**: [Open In Colab](https://colab.research.google.com/drive/1IuMvh0sCHniIwgHHNCUNXeH3SWuscEp0?usp=sharing)
Contains an end to end pipeline to run simulated test on the model. Currently it is performing averagely.

We then iterated over a number of pre-trained CNN to serve as feature extracts but amongst the lot, vgg16 performed very well on the ImageNet weights.
Finally, we clipped off the Multi-Linear Perceptron classifier layer and built a sequential model with regularized dense and units to enable us train a multi-class classifier.

## Bottleneck Features
To make this iteration reproducible, we have provided access to the bottle-necked features for the train, test and validation splits.
We finally trained and tested our model which has a far better variance compared to the other approaches used in the first two iterations and notebooks.
Per class accuracy metrics were evaluated on the model with the essential classes performing in a standard capacity.
The bottle neck features extracted from the vgg16 model include :

* Train
[train_data.pkl](https://github.com/LISA-Ghana/forest_guard_ai/blob/master/pickle_files/train_data.pkl)  
[train_labels.pkl](https://github.com/LISA-Ghana/forest_guard_ai/blob/master/pickle_files/train_labels.pkl)

* Test
[test_data.pkl](https://github.com/LISA-Ghana/forest_guard_ai/blob/master/pickle_files/test_data.pkl)  
[test_labels.pkl](https://github.com/LISA-Ghana/forest_guard_ai/blob/master/pickle_files/test_data.pkl)

* Validation
[validation_data.pkl](https://github.com/LISA-Ghana/forest_guard_ai/blob/master/pickle_files/validation_data.pkl)  
[validation_labels.pkl](https://github.com/LISA-Ghana/forest_guard_ai/blob/master/pickle_files/validation_labels.pkl)

## Models
1. [illegal_logging_classifier_model.h5](https://github.com/LISA-Ghana/forest_guard_ai/blob/master/Models/illegal_logging_classifier_model.h5) (Keras variant)
2. [ForestAI.tflite](https://github.com/LISA-Ghana/forest_guard_ai/blob/master/models/ForestAI.tflite) (Tensorflow-Lite variant)

We provide a converted/optimized TFLite format as well as an h5 Keras format of our model.
There is an inference pipeline script at the end of the notebook to enable one test the model on their audio files.

## Mobile App Client
The android apk can be found in the `apk/forest_guard_client.apk` folder. You can click [here](https://github.com/LISA-Ghana/forest_guard_ai/raw/master/apk/forest_guard_client.apk) to download.  
After installing the app, log in with `AgentID: 1234` and `Forest ID: 1`.  
When the prediction is sent to the database, deployed cloud functions trigger notifications for Human/Illegal (Chainsaw, etc) activity.

## Team Members  
  
| Name | Role | Profile |  
| :--- | :--- | :--- |  
| Appau Ernest (Lead) | AI Model Development/Deployment/I-IOT | [Github](https://github.com/kappernie) / [LinkedIn](https://www.linkedin.com/in/appauernestkofimensah/) |  
| Debrah Kwesi Buabeng | Mobile Developer | [Github](https://github.com/Akora-IngDKB) / [LinkedIn](https://www.linkedin.com/in/kwesi-buabeng-debrah) |  
| Akpalu Larry | Data Collection and Annotation | [Github](https://github.com/larry2310) / [LinkedIn](https://www.linkedin.com/in/larry-akpalu-5b3a1a119/) |  
| Kpene Godsway Edem | Documentation | [Github](https://github.com/kpegods96) / [LinkedIn](https://www.linkedin.com/in/godsway-edem-kpene-a6542711a/) |  
| Baidoo Mabel | Data Collection and Annotation | [Github](https://github.com/GeekiAdubea) / [LinkedIn](https://www.linkedin.com/in/mabel-adubea-baidoo/) |  
| Appau Roberta | UI/UX Designer | N/A |  
<br>

> We would like to express our sincere gratitude to all members of Team Lisa as well as  the mentors, host of this competition for being resourceful in our journey to seeing this through.

## References
* [Hands-On Mathematics for Deep Learning - Packt publishing](https://www.packtpub.com/product/hands-on-mathematics-for-deep-learning/9781838647292)
* [Environmental Sound Classification With Convolutional Neural Networks - Karol Piczak, 2015](https://www.karolpiczak.com/papers/Piczak2015-ESC-ConvNet.pdf)
* [Deep Learning - Ian Goodfellow and Yoshua Bengio](https://www.deeplearningbook.org/)
* [Deep learning with Keras workshop - Packt Publishing](https://courses.packtpub.com/courses/deep-learning-with-keras)
* [TinyML - O'Reilly](https://www.oreilly.com/library/view/tinyml/9781492052036/)
* [Keras](https://keras.io/)
* [Tensorflow](https://www.tensorflow.org/)
* Handouts by Superfluid Labs and AI4Good
* Mentors from Superfluid labs and AI4GOOD
* Google Search
* Wikipedia
