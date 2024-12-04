import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu

# Assuming you already have a DataFrame `df`
# Load the lists from your DataFrame
import pandas as pd

df = pd.read_csv('result.csv')
references = df['Answer'].to_list()
llama_food_answer = df['llama_food_answer'].to_list()
llama_answer_knowledge = df['llama_answer_knowledge'].to_list()
llama_answer = df['llama_answer'].to_list()

# Assuming references are a list of 20 sentences, where each element is a list of words
# Convert the reference sentences into a list of tokenized words (assuming they're stored as strings)
references = [ref.split() for ref in references]

# Function to calculate BLEU score for each method against the reference list
def calculate_bleu_score(reference_list, candidate_list):
    bleu_scores = []
    for ref, cand in zip(reference_list, candidate_list):
        cand_tokens = cand.split()  # Tokenizing candidate answer
        bleu_score = sentence_bleu([ref], cand_tokens)  # Calculate BLEU score
        bleu_scores.append(bleu_score)
    return bleu_scores

# Calculate BLEU scores for each method
bleu_method1 = calculate_bleu_score(references, llama_food_answer)
bleu_method2 = calculate_bleu_score(references, llama_answer_knowledge)
bleu_method3 = calculate_bleu_score(references, llama_answer)

# Print average BLEU score for each method
average_bleu_method1 = sum(bleu_method1) / len(bleu_method1)
average_bleu_method2 = sum(bleu_method2) / len(bleu_method2)
average_bleu_method3 = sum(bleu_method3) / len(bleu_method3)

print(f"Average BLEU Score for Llama Food Answer: {average_bleu_method1:.4f}")
print(f"Average BLEU Score for Llama Answer Knowledge: {average_bleu_method2:.4f}")
print(f"Average BLEU Score for Llama Answer: {average_bleu_method3:.4f}")

# Determine which method is closer to a reference BLEU score of 20
reference_bleu_score = 20
diff_method1 = abs(average_bleu_method1 - reference_bleu_score)
diff_method2 = abs(average_bleu_method2 - reference_bleu_score)
diff_method3 = abs(average_bleu_method3 - reference_bleu_score)

if diff_method1 < diff_method2 and diff_method1 < diff_method3:
    print("Llama Food Answer is closest to the reference BLEU score.")
elif diff_method2 < diff_method1 and diff_method2 < diff_method3:
    print("Llama Answer Knowledge is closest to the reference BLEU score.")
else:
    print("Llama Answer is closest to the reference BLEU score.")
