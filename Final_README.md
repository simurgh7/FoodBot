The weights of trained image classification models are available in this [link](https://usf.box.com/s/uda1dmwcqzpz6gyhz3jac6shfhypu5sc)
The test images of ten category of foods can also be found in this folder [link](https://usf.box.com/s/kf7f5a3sb2d3rdxsvgdg5t2gou4fn5na)
The weights of trained chat bot models are available in this [link](https://usf.box.com/s/5559jnyiq013a0dqfj4255jxcws6zs1a)
The weights of trained knowledge base models are available in this [link](https://usf.box.com/s/5559jnyiq013a0dqfj4255jxcws6zs1a)

Chatbot evaluation:

cd chatbot_evaluation

generate answer: evaluation_food.ipynb

use bleurt score to compare the generated answer vs true label answer

python bleurt_eval.py

use blue score to compare the generated answer vs true label answer

python blue_eval.py 

use rouge score to compare the generated answer vs true label answer

python rouge_eval.py 
