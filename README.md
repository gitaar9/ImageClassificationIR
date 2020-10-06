# ImageClassificationIR
This repo implements SVM classification based on ResNet feature extractors, like in https://www.researchgate.net/publication/333409943_Classification_of_Marine_Vessels_with_Multi-Feature_Structure_Fusion<br>
The classifiers can get up to 67.57% accuracy on the infrared data and 86.00% on the RGB data from the VAIS dataset(https://www.cv-foundation.org/openaccess/content_cvpr_workshops_2015/W05/papers/Zhang_VAIS_A_Dataset_2015_CVPR_paper.pdf)<br>
Also implements most of the preprocessing methods mentioned in https://ies.anthropomatik.kit.edu/ies/download/publ/ies_2018_hermann_infrared_person_detection.pdf


# Installing this repo
1. Git clone this repository
2. Create a venv and activate it
3. pip install -r requirements.txt
4. pip install -e .
