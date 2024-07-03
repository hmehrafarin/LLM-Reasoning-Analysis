import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create the DataFrame from the provided table
data = {
    "MODEL": ["LLaMA 2 13b", "LLaMA 2 13b", "LLaMA 7b", "LLaMA 7b", "Flan T5", "Flan T5"],
    "EXPERIMENT": ["Full", "both facts shuffled", "Full", "both facts shuffled", "Full", "both facts shuffled"],
    "SCORE": [90, 86, 74, 73, 97, 92]
}

df = pd.DataFrame(data)

# Create the improved bar chart
fig, ax = plt.subplots(figsize=(10, 6))

# Set a seaborn style
sns.set(style="whitegrid")

# Plot the data with seaborn color palette
colors = sns.color_palette("Set2")

bar_width = 0.35
index = range(len(df[df["EXPERIMENT"] == "Full"]))

bar1 = ax.bar(index, df[df["EXPERIMENT"] == "Full"]["SCORE"], bar_width, label='Full', color="#15616d")
bar2 = ax.bar([i + bar_width for i in index], df[df["EXPERIMENT"] == "both facts shuffled"]["SCORE"], bar_width, label='both facts shuffled', color="#ff7d00")

# Add labels, title, and legend with improved aesthetics
# ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=18)
# ax.set_title('Model Scores by Experiment', fontsize=15)
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(df[df["EXPERIMENT"] == "Full"]["MODEL"], fontsize=18)
ax.yaxis.set_tick_params(labelsize=16)
ax.legend(fontsize=16)


# Add score labels on the bars with improved aesthetics
for bar in bar1:
    height = bar.get_height()
    ax.annotate('{}'.format(height),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=16)

for bar in bar2:
    height = bar.get_height()
    ax.annotate('{}'.format(height),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=16)

# Remove top and right spines
sns.despine()

plt.show()
plt.savefig("model_scores.png")