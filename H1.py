import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# ==========================================
# 1. DATA LOADING & FEATURE ENGINEERING
# ==========================================

# Load music genre data
music_df = pd.read_csv("cleaned.csv")
if 'Unnamed: 0' in music_df.columns:
    music_df = music_df.drop(columns=['Unnamed: 0'])

genre_cols = ['Hip hop/Rap/R&b', 'EDM', 'Pop', 'Rock/Metal', 'Latin/Reggaeton', 'Other']

# Calculate Shannon Diversity Index for musical genres
def calculate_diversity(row):
    total = row.sum()
    if total == 0: return 0
    props = row / total
    props = props[props > 0] # Filter to avoid log(0)
    return -np.sum(props * np.log(props))

# Apply the function to create the new feature
music_df['Diversity_Index'] = music_df[genre_cols].apply(calculate_diversity, axis=1)
music_df['Country'] = music_df['Country'].str.lower().str.strip()

# ==========================================
# 2. DATA INTEGRATION (ENRICHMENT)
# ==========================================

# Load happiness data (using 2019 as the primary cross-sectional year)
hap_df = pd.read_csv("2019.csv")
hap_df = hap_df.rename(columns={'Country or region': 'Country', 'Score': 'Happiness_Score'})
hap_df['Country'] = hap_df['Country'].str.lower().str.strip()

# Merge datasets on Country
merged_df = pd.merge(music_df, hap_df[['Country', 'Happiness_Score']], on='Country', how='inner')

# ==========================================
# 3. HYPOTHESIS TESTING (STATISTICAL ANALYSIS)
# ==========================================

# Calculate Pearson's Correlation Coefficient and p-value
pearson_coef, p_value = stats.pearsonr(merged_df['Diversity_Index'], merged_df['Happiness_Score'])

print("-" * 50)
print("HYPOTHESIS TESTING RESULTS")
print("-" * 50)
print(f"H0: There is no correlation between Musical Diversity and Happiness.")
print(f"Ha: There is a positive correlation between Musical Diversity and Happiness.\n")
print(f"Pearson Correlation Coefficient (r): {pearson_coef:.4f}")
print(f"P-value: {p_value:.4f}\n")

if p_value < 0.05:
    print("CONCLUSION: REJECT Null Hypothesis (H0).")
    print("Evidence supports that countries with higher musical diversity report higher happiness scores.")
else:
    print("CONCLUSION: FAIL TO REJECT Null Hypothesis (H0).")
print("-" * 50)

# ==========================================
# 4. VISUALIZATION
# ==========================================

sns.set_theme(style="whitegrid")
plt.figure(figsize=(9, 6))

# Create regression plot
sns.regplot(data=merged_df, 
            x='Diversity_Index', 
            y='Happiness_Score', 
            scatter_kws={'alpha': 0.7, 'color': '#2980b9'}, 
            line_kws={'color': '#e74c3c', 'linewidth': 2})

plt.title("Hypothesis Test: Music Diversity vs. National Happiness", fontsize=14, fontweight='bold')
plt.xlabel("Shannon Diversity Index (Musical Variety)", fontsize=12)
plt.ylabel("National Happiness Score", fontsize=12)

# Annotate the plot with statistical findings
annot_text = f"r = {pearson_coef:.3f}\np = {p_value:.4f}"
plt.text(0.05, 0.95, annot_text, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()

# Save the figure for the final report
# plt.savefig("hypothesis_test_result.png", dpi=300)
plt.show()