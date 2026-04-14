import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# ==========================================
# 1. DATA PREPARATION & FEATURE ENGINEERING
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

music_df['Diversity_Index'] = music_df[genre_cols].apply(calculate_diversity, axis=1)
music_df['Country'] = music_df['Country'].str.lower().str.strip()

# ==========================================
# 2. DATA ENRICHMENT (MERGING WITH GDP)
# ==========================================

# Load World Happiness Report data (using 2019) to extract GDP
hap_df = pd.read_csv("2019.csv")
hap_df = hap_df.rename(columns={'Country or region': 'Country'})
hap_df['Country'] = hap_df['Country'].str.lower().str.strip()

# Merge datasets based on Country
merged_df = pd.merge(music_df, hap_df[['Country', 'GDP per capita']], on='Country', how='inner')

# ==========================================
# 3. HYPOTHESIS 2 TESTING
# ==========================================

print("-" * 60)
print("HYPOTHESIS 2: GDP PER CAPITA VS. MUSICAL DIVERSITY")
print("-" * 60)
print("H0: There is no correlation between GDP per capita and Musical Diversity.")
print("Ha: There is a positive correlation between GDP per capita and Musical Diversity.\n")

# Calculate Pearson's Correlation Coefficient and p-value
pearson_coef, p_value = stats.pearsonr(merged_df['GDP per capita'], merged_df['Diversity_Index'])

print(f"Pearson Correlation Coefficient (r): {pearson_coef:.4f}")
print(f"P-value: {p_value:.4f}\n")

if p_value < 0.05:
    print("CONCLUSION: REJECT Null Hypothesis (H0).")
    print("Evidence supports that as economic wealth increases, populations tend to explore a more diverse range of musical genres.")
else:
    print("CONCLUSION: FAIL TO REJECT Null Hypothesis (H0).")
print("-" * 60)

# ==========================================
# 4. VISUALIZATION
# ==========================================

sns.set_theme(style="whitegrid")
plt.figure(figsize=(9, 6))

# Create regression plot
sns.regplot(data=merged_df, 
            x='GDP per capita', 
            y='Diversity_Index', 
            scatter_kws={'alpha': 0.7, 'color': '#27ae60', 's': 60}, 
            line_kws={'color': '#c0392b', 'linewidth': 2})

plt.title("Hypothesis 2: Economic Wealth (GDP) vs. Music Diversity", fontsize=14, fontweight='bold')
plt.xlabel("GDP per Capita", fontsize=12)
plt.ylabel("Shannon Diversity Index", fontsize=12)

# Annotate the plot with statistical findings
annot_text = f"r = {pearson_coef:.3f}\np = {p_value:.4f}"
plt.text(0.05, 0.95, annot_text, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.show()