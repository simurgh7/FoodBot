import numpy as np
from bleurt import score
import pandas as pd

df = pd.read_csv('result.csv')
# Load BLEURT model
bleurt_checkpoint = "../BLEURT-20"  # Replace with your BLEURT model checkpoint
scorer = score.BleurtScorer(bleurt_checkpoint)

# Sample references and outputs
references = df.Answer.to_list()

llama_food_answer = df.llama_food_answer.to_list()

llama_answer_knowledge = df.llama_answer_knowledge.to_list()
llama_answer = df.llama_answer.to_list()

# Calculate BLEURT scores
scores_method_1 = scorer.score(references=references, candidates=llama_food_answer)
scores_method_2 = scorer.score(references=references, candidates=llama_answer_knowledge)
scores_method_3 = scorer.score(references=references, candidates=llama_answer)

# Calculate average scores
avg_score_method_1 = np.mean(scores_method_1)
avg_score_method_2 = np.mean(scores_method_2)
avg_score_method_3 = np.mean(scores_method_3)

# Compare results
print(f"llama_food_answer Average BLEURT Score: {avg_score_method_1:.4f}")
print(f"llama_answer_knowledge Average BLEURT Score: {avg_score_method_2:.4f}")
print(f"llama_answer Average BLEURT Score: {avg_score_method_3:.4f}")

# if avg_score_method_1 > avg_score_method_2:
#     print("Method 1 performs better.")
# else:
#     print("Method 2 performs better.")
