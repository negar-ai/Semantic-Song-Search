import pandas as pd
from search import search_by_text
import os

def load_eval_set(path = 'evaluation/eval_queries.csv'):
    df = pd.read_csv(path, encoding='utf-8')
    eval_set = []
    for _, row in df.iterrows():
        relevant = set(t.strip() for t in row['relevant_titles'].split('|'))
        eval_set.append({'query': row['query'], 'relevant_titles': relevant})
    return eval_set

def precision_at_k(retrieved_titles, relevant_titles, k=5):
    top_k = retrieved_titles[:k]
    hits = sum(1 for t in top_k if t in relevant_titles)
    return hits / k

def recall_at_k(retrieved_titles, relevant_titles, k=5):
    top_k = retrieved_titles[:k]
    hits = sum(1 for t in top_k if t in relevant_titles)
    return hits / len(relevant_titles)

def evaluate(eval_set, k = 5, verbose = True):
    precisions, recalls = [], []
    for item in eval_set:
        result = search_by_text(item['query'], k=k)
        retrieved_titles = [r['Title'] for r in result['results']]
        p = precision_at_k(retrieved_titles, item['relevant_titles'], k = k)
        r = recall_at_k(retrieved_titles, item['relevant_titles'], k = k)
        precisions.append(p)
        recalls.append(r)

        if verbose:
            print(f"\nQuery: {item['query']}")
            print(f"  Retrieved: {retrieved_titles}")
            print(f"  Relevant:  {item['relevant_titles']}")
            print(f"  Precision@{k}: {p:.2f}  Recall@{k}: {r:.2f}")

    mean_precision = sum(precisions) / len(precisions)
    mean_recall = sum(recalls) / len(recalls)
    print(f"\n{'='*40}\nMean Precision@{k}: {mean_precision:.3f}")
    print(f"Mean Recall@{k}: {mean_recall:.3f}\n{'='*40}")
    return {"mean_precision": mean_precision, "mean_recall": mean_recall}

if __name__ == "__main__":
    eval_set = load_eval_set()
    evaluate(eval_set, k=20)