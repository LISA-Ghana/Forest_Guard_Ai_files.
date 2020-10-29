# Aicommons - Ai4good competition (Edge based early detection and alerting systems for forest-Guards) submission by team Lisa
# Project Write-Up

## Repo Tree

	
	├───apk
		└───forest_guard_client.apk

	├───labels_and_classes-csv
		├───ESC-50_Mel-Spectogram_dataset_Meta_data.csv
		├───ESC-50_WavePlot_dataset_Meta_data.csv
		└───labels_and_target_classes(ESC50).csv
	
	├───Models
		├───ForestAI.tflite
		└───illegal_logging_classifier_model.h5
	
	├───Notebooks
		├───Dataset_exploration_and__Generation_by_Author_Appau_Ernest.ipynb
		├───iteration1_Building_a_CNN_model_to_classify_audio_events_from_the_ESC_50_WavePlot_dataset.ipynb
		└───Iteration2_Building_a_CNN_model_to_classify_audio_events_from_the_ESC_50_MelSpectogram_dataset.ipynb

	├───Pickle files
		├───test_data.pkl
		├───test_labels.pkl
		├───train_labels.pkl
		├───train_data.pkl
		├───validation_labels.pkl
		└───validation_data.pkl
		
	├───forest-guard-ai4good-firebase-adminsdk-1fw9p-232da17716.json
	
	└───Testing_final_model_Team Lisa.ipynb




#### click the links attached  `Testing_final_notebook`  to make a copy of this Notebook to run sample test in colab ,it contains an end to end pipeline with instructions to run inference on sound inputs
####  Testing_final_notebook -[link to colab Notebook ](https://colab.research.google.com/drive/1IuMvh0sCHniIwgHHNCUNXeH3SWuscEp0?usp=sharing) 

## Datasets used -
	 Primary:Esc-50 dataset:- https://github.com/karoldvl/ESC-50/archive/master.zip 
	 Secondary dataset for sampled audio for testing purposes:-https://research.google.com/audioset

## Datasets (2)variants generated  -
	 i-Esc-50 waveplot/amplitude portfolio dataset
	 ii-ESC-50 MelSpectogram portfolio dataset

# This Repo contains five notebooks:

#### Notebook_1:-[link to colab Notebook ](https://colab.research.google.com/drive/1cUb_4mp9zQujwEs75m7ZVRKIzXieBqsq?usp=sharing)
This jupyter notebook involves a feature engineering approach to generate two distinct datasets
to help us in our modelling process in an attempt to solve the challenge of illgal logging activities
Generated Datasets include:
	1-Esc-50 waveplot dataset
	2-Esc-50 mel spectogram dataset

#### Notebook_2 :-[link to colab Notebook ](https://colab.research.google.com/drive/1seJqPEI2YLDNASsPrShS85QRSKIxrHvK?usp=sharing)
our first iteration by building a base convolutional neral network to classify 
audio waveplot/amplitude profile.
Dataset used in this notebook includes Generated ESC-50 waveplot profiles of the original 
audio files.
Results from this iteration was poor performing model with a high bias and variance between 
training and validation splits.In conclusion, we moved on to the melspectogram dataset to achieve a better 
performance after dozen hyperparameter tuning exercises


#### Notebook_3 :-[link to colab Notebook ](https://colab.research.google.com/drive/1lN8Z5dbJYTT8Y531shL3YyFTEe93H1Kf?usp=sharing)
In this notebook, our model perfomed a bit better than our previous data used in notebook 2 but also displayed 
a high level of variance during training process.After a dozen hyper parameter tuning and regularization,we concluded 
on the lack of our generated dataset not well feature engineerd enough to accomodate a genearlized model


#### Notebook 4 :-[link to colab Notebook ](https://colab.research.google.com/drive/1p7iOe5fQz6TeK1YUWbf4A5CbkgsSdn_S?usp=sharing)
we adopted an advanced feature engineering approach by converting our audio files to a concatenated stack of 3 channels
based on three main operations to generate feature maps which includes spliting the 5 second audio clips into 5 folds making our intial dataset
increase from 2000 audio files to 10000 files.we  combined three feature maps which include the Mel spectogram,log scaled spectogram and the delta melscaled
spectogram to form one feature map profile for an audio clip

#### Notebook 5 :- [link to colab Notebook ](https://colab.research.google.com/drive/1IuMvh0sCHniIwgHHNCUNXeH3SWuscEp0?usp=sharing)

Contains an end to end pipeline to run simulated test on the model, currently it is performing averagely.More finetuning needed

--------------------------------------------------------------------------------------------------------------------------

We then iterated over a number of pre-trained convolutional neural networks to serve as feature extractos but amongs the lot, vgg16 performed very well on the imagent weights 
and finally , we clipped off the muliti linear perceptron classifier layer and architected a sequential model with regularized dense and units to enable us train a calssifier
.
### bottleneck Features
To make this iteration reproducible, we have provided access to the bottlenecked features for the train,test and validation splits.
we finally train and test our model which has a far better variance compared to the other approaches used in the first two iterations and notebooks.
Per class accuracy metrics were evaluated on the model with the essential classes performig in a standard capacity.
The bottle neck features extracted from the vgg16 model include :

#####  Train
	  train_data.pkl
	  train_labels.pkl
##### Test
	  test_data.pkl
	  test_labels.pkl
#####  Validation
	 validation_data.pkl
	 validation_labels.pkl

## Models in this repo include:
	 1-illegal_logging_classifier_model.h5(Keras variant)
	 2-ForestAI.tflite (tensorflowlite variant)

we provide a converted/optimized tensorflowlite format of our model as well as a h5 format of our model 
with a inference pipline script at the end of the notebook to enable one test the model on their audio files 

# Mobile App Client
The android apk can be found in the `apk/forest_guard_client.apk` folder. You can click [here](https://github.com/LISA-Ghana/forest_guard_ai/raw/master/apk/forest_guard_client.apk) to download.  
After installing the app, log in with `AgentID: 1234` and `Forest ID: 1`.  
When the prediction is sent to the database, deployed cloud functions trigger notifications for Human/Illegal (Chainsaw, etc) activity.

## Team Members  
  
| Name | Role | Profile |  
| :--- | :--- | :--- |  
| Appau Ernest (Lead) | AI Model Development/Deployment | [Github](https://github.com/kappernie) / [LinkedIn](https://www.linkedin.com/in/appauernestkofimensah/) |  
| Debrah Kwesi Buabeng | Mobile Developer | [Github](https://github.com/Akora-IngDKB) / [LinkedIn](https://www.linkedin.com/in/kwesi-buabeng-debrah) |  
| Akpalu Larry | Data Collection and Annotation | [Github](https://github.com/larry2310) / [LinkedIn](https://www.linkedin.com/in/larry-akpalu-5b3a1a119/) |  
| Kpene Godsway Edem | Documentation | [Github](https://github.com/kpegods96) / [LinkedIn](https://www.linkedin.com/in/godsway-edem-kpene-a6542711a/) |  
| Baidoo Mabel | Data Collection and Annotation | [Github](https://github.com/GeekiAdubea) / [LinkedIn](https://www.linkedin.com/in/mabel-adubea-baidoo/) |  
| Appau Roberta | UI/UX Designer |  |  

<br>
we would like to express our sincere gratitude to the members of team Lisa as well as  the 
mentors ,host of this competition for being resourceful in our journey to seeing this through


## References
HANDS ON mathemathics for deep learning algorithms -Packt publishing,
Environmental Sound classification Paper 2015 by Karol Piczak,
Deep learning book by Ian Goodfellow and Yoshua Bengio,
Deep learning with Keras workshop by Packt Publishing,
Google search,
Wikipaedia,
handouts by Superfluid labs and Ai4good meeetups,
Mentors from Superfluid labs and AI4GOOD,
https://keras.io/api/applications/,
keras.io,
TinyMl book,
Tensorflow.org
Tinyml book by Peter Warden

