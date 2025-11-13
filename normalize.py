import argparse
import pandas as pd

# Example code to calculate composite score
metrics = ['METEOR', 'Rouge-1.f', 'Rouge-2.f', 'Rouge-l.f', 'BLEU', 'Laplace Perplexity', 'Lidstone Perplexity', 'Cosine similarity', 'Pearson correlation', 'F1 score', 'Bert-Score.precision', 'Bert-Score.recall', 'Bert-Score.f1', 'B-RT.coherence', 'B-RT.consistency', 'B-RT.fluency', 'B-RT.relevance', 'B-RT.average']
weights = {'METEOR': 0.15, 'Rouge-1.f': 0.0, 'Rouge-2.f': 0.075, 'Rouge-l.f': 0.075, 'BLEU': 0.0, 'Laplace Perplexity': 0.1, 'Lidstone Perplexity': 0.1, 'Cosine similarity': 0.0, 'Pearson correlation': 0.0, 'F1 score': 0.15, 'Bert-Score.precision': 0.0, 'Bert-Score.recall': 0.0, 'Bert-Score.f1': 0.125, 'B-RT.coherence': 0.0, 'B-RT.consistency': 0.0, 'B-RT.fluency': 0.10, 'B-RT.relevance': 0.0, 'B-RT.average': 0.125}

def normalize(series):
   return (series - series.min()) / (series.max() - series.min())

def composite_score(df, weights):
   normalized_df = df[metrics].apply(normalize)
   print("IN COMPOSITE SCORE\n", normalized_df)
   score = sum(normalized_df[metric] * weight for metric, weight in weights.items())
   print("SCORE in DEF\n", score)
   return score

parser = argparse.ArgumentParser(description='Calculate normalized scores.')
parser.add_argument('file', help='The Excel file to read.')
args = parser.parse_args()

df = pd.read_excel(args.file, engine='openpyxl')

df = pd.read_excel(args.file)
df_selected = df.iloc[0:372, 5:29]

df_to_normalize = df.iloc[2:372, 6:29]
normalized_df = df_to_normalize.apply(normalize)
df.iloc[2:372, 6:29] = normalized_df

#print(df)

# Calculate the composite score
composite_score = composite_score(df_selected, weights)
print ("COMPOSITE SCORE\n", composite_score, composite_score.mean())
