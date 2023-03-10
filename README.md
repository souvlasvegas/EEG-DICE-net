# Alzheimers Project
<h1>Transformer NN Alzheimer's</h1>
    
In this project, a Dual Input Convolution Encoder Network (DICE-net) model is created for the classification of EEG signals of Alzheimer's patients for the paper:

<em>to be filled when published</em>

The project is consisted of the packages: 1) feature_extraction, 2) machine_learning.
<br>The folder paper_related_code was for visualization purposes, not structured and you should ignore it

<h2>Feature Extraction</h2>

>The preprocessing was performed in EEGLAB matlab. Save files in .set format and run dataset_creation.py to create dataset. The whole process is dependent to the naming of the files (in our case, .set file was named as A1.set, A2.set, C5.set etc). If you want to use it with other naming system, please modify accordingly. To reproduce, create environment based on environment.yml file.
><br>
><br>This package creates dataset of extracted Relative Band Power and Spectral Coherence Connectivity features. Check docstring documentation for more information.

<h2>Machine Learning</h2>

>The model was implemented in PyTorch. To reproduce, create environment based on environment.yml file (different from the Feature Extraction yml).
><br>
><br>The model takes a .pkl file as input. To reproduce, use AlzheimerTrainingDatasetDemo.pkl. Full dataset cannot be uploaded due to Ethics Restrictions. For full documentation of the model, go to the published journal article.