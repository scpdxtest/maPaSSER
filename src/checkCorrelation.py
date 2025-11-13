import numpy as np
from scipy import stats
import pandas as pd
import sys
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Check if we have the right number of arguments
if len(sys.argv) != 5:
    print("Usage: python script.py data_name1 data_name2 data_name3 lines")
    sys.exit(1)

# Read the arguments
data_name1 = sys.argv[1]
data_name2 = sys.argv[2]
data_name3 = sys.argv[3]
how_many_lines = int(sys.argv[4])

# Load the workbook
df1 = pd.read_excel(data_name1)
df2 = pd.read_excel(data_name2)
df3 = pd.read_excel(data_name3)

# calculate correlation matrix for df1
data_cor5_1 = df1.iloc[1:how_many_lines, 5].tolist()
data_cor6_1 = df1.iloc[1:how_many_lines, 6].tolist()
data_cor7_1 = df1.iloc[1:how_many_lines, 7].tolist()
data_cor8_1 = df1.iloc[1:how_many_lines, 8].tolist()
data_cor12_1 = df1.iloc[1:how_many_lines, 12].tolist()
data_cor13_1 = df1.iloc[1:how_many_lines, 13].tolist()
data_cor14_1 = df1.iloc[1:how_many_lines, 14].tolist()
data_cor15_1 = df1.iloc[1:how_many_lines, 15].tolist()
data_cor16_1 = df1.iloc[1:how_many_lines, 16].tolist()
data_cor17_1 = df1.iloc[1:how_many_lines, 17].tolist()
data_cor18_1 = df1.iloc[1:how_many_lines, 18].tolist()
data_cor19_1 = df1.iloc[1:how_many_lines, 19].tolist()
data_cor20_1 = df1.iloc[1:how_many_lines, 20].tolist()

# calculate correlation matrix for df2
data_cor5_2 = df2.iloc[1:how_many_lines, 5].tolist()
data_cor6_2 = df2.iloc[1:how_many_lines, 6].tolist()
data_cor7_2 = df2.iloc[1:how_many_lines, 7].tolist()
data_cor8_2 = df2.iloc[1:how_many_lines, 8].tolist()
data_cor12_2 = df2.iloc[1:how_many_lines, 12].tolist()
data_cor13_2 = df2.iloc[1:how_many_lines, 13].tolist()
data_cor14_2 = df2.iloc[1:how_many_lines, 14].tolist()
data_cor15_2 = df2.iloc[1:how_many_lines, 15].tolist()
data_cor16_2 = df2.iloc[1:how_many_lines, 16].tolist()
data_cor17_2 = df2.iloc[1:how_many_lines, 17].tolist()
data_cor18_2 = df2.iloc[1:how_many_lines, 18].tolist()
data_cor19_2 = df2.iloc[1:how_many_lines, 19].tolist()
data_cor20_2 = df2.iloc[1:how_many_lines, 20].tolist()

# calculate correlation matrix for df3
data_cor5_3 = df3.iloc[1:how_many_lines, 5].tolist()
data_cor6_3 = df3.iloc[1:how_many_lines, 6].tolist()
data_cor7_3 = df3.iloc[1:how_many_lines, 7].tolist()
data_cor8_3 = df3.iloc[1:how_many_lines, 8].tolist()
data_cor12_3 = df3.iloc[1:how_many_lines, 12].tolist()
data_cor13_3 = df3.iloc[1:how_many_lines, 13].tolist()
data_cor14_3 = df3.iloc[1:how_many_lines, 14].tolist()
data_cor15_3 = df3.iloc[1:how_many_lines, 15].tolist()
data_cor16_3 = df3.iloc[1:how_many_lines, 16].tolist()
data_cor17_3 = df3.iloc[1:how_many_lines, 17].tolist()
data_cor18_3 = df3.iloc[1:how_many_lines, 18].tolist()
data_cor19_3 = df3.iloc[1:how_many_lines, 19].tolist()
data_cor20_3 = df3.iloc[1:how_many_lines, 20].tolist()

# Calculate the correlation
correlation_matrix1 = np.corrcoef([data_cor5_1, data_cor6_1, data_cor7_1, data_cor8_1, data_cor12_1, data_cor13_1, data_cor14_1, data_cor15_1, data_cor16_1, data_cor17_1, data_cor18_1, data_cor19_1, data_cor20_1])
correlation_matrix2 = np.corrcoef([data_cor5_2, data_cor6_2, data_cor7_2, data_cor8_2, data_cor12_2, data_cor13_2, data_cor14_2, data_cor15_2, data_cor16_2, data_cor17_2, data_cor18_2, data_cor19_2, data_cor20_2])
correlation_matrix3 = np.corrcoef([data_cor5_3, data_cor6_3, data_cor7_3, data_cor8_3, data_cor12_3, data_cor13_3, data_cor14_3, data_cor15_3, data_cor16_3, data_cor17_3, data_cor18_3, data_cor19_3, data_cor20_3])

# print("Correlation matrix for " + data_name1 + ":\n", correlation_matrix1)
# print("\nCorrelation matrix for " + data_name2 + ":\n", correlation_matrix2)
# print("\nCorrelation matrix for " + data_name3 + ":\n", correlation_matrix3)

resTitles = ['METEOR', 
            'Rouge-1.r', 'Rouge-1.p', 'Rouge-1.f',
            'Rouge-l.r', 'Rouge-l.p', 'Rouge-l.f',
            'BLUE', 'Laplace Perplexity', 'Lidstone Perplexity',
            'Cosine similarity', 'Pearson correlation', 'F1 score']
df_resTitles = pd.DataFrame([resTitles])

df1_out = pd.DataFrame(correlation_matrix1, columns=resTitles)
df2_out = pd.DataFrame(correlation_matrix2, columns=resTitles)
df3_out = pd.DataFrame(correlation_matrix3, columns=resTitles)

df1_out.to_excel('correlation_matrix_' + data_name1 + '.xlsx', index=False)
df2_out.to_excel('correlation_matrix_' + data_name2 + '.xlsx', index=False)
df3_out.to_excel('correlation_matrix_' + data_name3 + '.xlsx', index=False)

print("Correlation matrix for " + data_name1 + " saved to correlation_matrix_" + data_name1 + ".xlsx")
print("Correlation matrix for " + data_name2 + " saved to correlation_matrix_" + data_name2 + ".xlsx")
print("Correlation matrix for " + data_name3 + " saved to correlation_matrix_" + data_name3 + ".xlsx")