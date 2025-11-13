import matplotlib.pyplot as plt

# Data for visualization
similarity_scores = ['0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95']
mistral_scores = [0.463150, 0.487886, 0.463460, 0.460408, 0.468361, 0.424785, 0.463281, 0.482175, 0.468986, 0.481841]
orca_scores = [0.483298, 0.488232, 0.505304, 0.526650, 0.500202, 0.472629, 0.518104, 0.526259, 0.521275, 0.491266]
llama_scores = [0.458887, 0.491925, 0.411094, 0.483548, 0.490521, 0.484744, 0.475556, 0.484952, 0.489074, 0.416642]

# Create a plot
plt.figure(figsize=(12, 8))

# Plot Mistral scores
plt.plot(similarity_scores, mistral_scores, marker='o', label='Mistral Composite Score')
# Plot Orca scores
plt.plot(similarity_scores, orca_scores, marker='o', label='Orca Composite Score')
# Plot Llama scores
plt.plot(similarity_scores, llama_scores, marker='o', label='Llama Composite Score')

# Add title and labels
plt.title('Composite Scores for Mistral, Orca, and Llama LLMs Across Different Similarity Score Thresholds')
plt.xlabel('Similarity Score Threshold')
plt.ylabel('Composite Score')
plt.legend()
plt.grid(True)
plt.ylim(0.4, 0.55)

# Show plot
plt.show()