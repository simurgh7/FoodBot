## General Idea
In this project, we propose a chatbot built using Llama2, combined with our image classification model to identify and provide information about selected dishes. The image model will analyze an image of a dish and predict its name. This name will then be relayed to the chatbot, which will offer assistance regarding the dish's flavor, ingredients, and any allergy warnings. Our work provides an end-to-end solution for users seeking information about new dishes. It will help users gain general knowledge and be aware of potential allergy risks before ordering food at a new restaurant.

## Demo
In the video, we show a demo on how to use our product
![Step](midterm_demo.gif)

## Weights
The weights of trained image classification models are available in this [link](https://usf.box.com/s/uda1dmwcqzpz6gyhz3jac6shfhypu5sc).
The weights of trained chat bot models are available in this [link](https://usf.box.com/s/5559jnyiq013a0dqfj4255jxcws6zs1a)

## Test Data
The test images of ten category of foods can also be found in this folder [link](https://usf.box.com/s/kf7f5a3sb2d3rdxsvgdg5t2gou4fn5na)
The weights of trained knowledge base models are available in this [link](https://usf.box.com/s/5559jnyiq013a0dqfj4255jxcws6zs1a)
## Image Model Evaluation
The image models are evaluated based on two categories - evasion detection and evasion attack on model.
The poisoned images are stored in the given folder. 

Evasion examples are generated using - python swin_fgsm.py
The given inference codes can be utilized to test the evasion attacks using the generated samples. 

Evasion detection run - python swin_evasion_detector.py

## Chatbot Evaluation

cd chatbot_evaluation

generate answer: evaluation_food.ipynb

use bleurt score to compare the generated answer vs true label answer

python bleurt_eval.py

use blue score to compare the generated answer vs true label answer

python blue_eval.py 

use rouge score to compare the generated answer vs true label answer

python rouge_eval.py 
