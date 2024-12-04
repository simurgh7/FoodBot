import pandas as pd
from rouge_score import rouge_scorer

df = pd.read_csv('result.csv')

# Assuming df is your DataFrame
references = df.Answer.to_list()
llama_food_answer = df.llama_food_answer.to_list()
llama_answer_knowledge = df.llama_answer_knowledge.to_list()
llama_answer = df.llama_answer.to_list()

def compute_rouge_score(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores

def aggregate_rouge_scores(references, llama_food_answer, llama_answer_knowledge, llama_answer):
    # Initialize lists to store scores for all comparisons
    rouge1_scores_food = []
    rouge2_scores_food = []
    rougeL_scores_food = []
    
    rouge1_scores_knowledge = []
    rouge2_scores_knowledge = []
    rougeL_scores_knowledge = []
    
    rouge1_scores_answer = []
    rouge2_scores_answer = []
    rougeL_scores_answer = []
    
    # Iterate over all the examples and compute ROUGE scores
    for idx in range(len(references)):
        reference = references[idx]
        
        # Compare llama_food_answer with the reference
        scores_food = compute_rouge_score(reference, llama_food_answer[idx])
        rouge1_scores_food.append(scores_food['rouge1'].fmeasure)
        rouge2_scores_food.append(scores_food['rouge2'].fmeasure)
        rougeL_scores_food.append(scores_food['rougeL'].fmeasure)
        
        # Compare llama_answer_knowledge with the reference
        scores_knowledge = compute_rouge_score(reference, llama_answer_knowledge[idx])
        rouge1_scores_knowledge.append(scores_knowledge['rouge1'].fmeasure)
        rouge2_scores_knowledge.append(scores_knowledge['rouge2'].fmeasure)
        rougeL_scores_knowledge.append(scores_knowledge['rougeL'].fmeasure)
        
        # Compare llama_answer with the reference
        scores_answer = compute_rouge_score(reference, llama_answer[idx])
        rouge1_scores_answer.append(scores_answer['rouge1'].fmeasure)
        rouge2_scores_answer.append(scores_answer['rouge2'].fmeasure)
        rougeL_scores_answer.append(scores_answer['rougeL'].fmeasure)
    
    # Calculate the average scores for each method
    avg_rouge1_food = sum(rouge1_scores_food) / len(rouge1_scores_food)
    avg_rouge2_food = sum(rouge2_scores_food) / len(rouge2_scores_food)
    avg_rougeL_food = sum(rougeL_scores_food) / len(rougeL_scores_food)
    
    avg_rouge1_knowledge = sum(rouge1_scores_knowledge) / len(rouge1_scores_knowledge)
    avg_rouge2_knowledge = sum(rouge2_scores_knowledge) / len(rouge2_scores_knowledge)
    avg_rougeL_knowledge = sum(rougeL_scores_knowledge) / len(rougeL_scores_knowledge)
    
    avg_rouge1_answer = sum(rouge1_scores_answer) / len(rouge1_scores_answer)
    avg_rouge2_answer = sum(rouge2_scores_answer) / len(rouge2_scores_answer)
    avg_rougeL_answer = sum(rougeL_scores_answer) / len(rougeL_scores_answer)
    
    # Print the combined average ROUGE scores for each method
    print("Average ROUGE Scores for all comparisons:")
    print(f"llama_food_answer:")
    print(f"ROUGE-1: {avg_rouge1_food:.4f}, ROUGE-2: {avg_rouge2_food:.4f}, ROUGE-L: {avg_rougeL_food:.4f}")
    
    print(f"\nllama_answer_knowledge:")
    print(f"ROUGE-1: {avg_rouge1_knowledge:.4f}, ROUGE-2: {avg_rouge2_knowledge:.4f}, ROUGE-L: {avg_rougeL_knowledge:.4f}")
    
    print(f"\nllama_answer:")
    print(f"ROUGE-1: {avg_rouge1_answer:.4f}, ROUGE-2: {avg_rouge2_answer:.4f}, ROUGE-L: {avg_rougeL_answer:.4f}")
    
    # Optionally, compare methods
    print("\nComparison of methods based on average ROUGE scores:")
    if avg_rouge1_food > avg_rouge1_knowledge and avg_rouge1_food > avg_rouge1_answer:
        print("llama_food_answer performs best for ROUGE-1.")
    elif avg_rouge1_knowledge > avg_rouge1_food and avg_rouge1_knowledge > avg_rouge1_answer:
        print("llama_answer_knowledge performs best for ROUGE-1.")
    else:
        print("llama_answer performs best for ROUGE-1.")
    
    if avg_rouge2_food > avg_rouge2_knowledge and avg_rouge2_food > avg_rouge2_answer:
        print("llama_food_answer performs best for ROUGE-2.")
    elif avg_rouge2_knowledge > avg_rouge2_food and avg_rouge2_knowledge > avg_rouge2_answer:
        print("llama_answer_knowledge performs best for ROUGE-2.")
    else:
        print("llama_answer performs best for ROUGE-2.")
    
    if avg_rougeL_food > avg_rougeL_knowledge and avg_rougeL_food > avg_rougeL_answer:
        print("llama_food_answer performs best for ROUGE-L.")
    elif avg_rougeL_knowledge > avg_rougeL_food and avg_rougeL_knowledge > avg_rougeL_answer:
        print("llama_answer_knowledge performs best for ROUGE-L.")
    else:
        print("llama_answer performs best for ROUGE-L.")

# Example usage
aggregate_rouge_scores(references, llama_food_answer, llama_answer_knowledge, llama_answer)

