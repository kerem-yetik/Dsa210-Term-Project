Milestone 1: Exploratory Data Analysis & Hypothesis Testing
Project Overview
This project, titled "Harmonic Well-being: A Spatiotemporal Analysis of Music Diversity and National Happiness," investigates the relationship between global music consumption patterns and socio-economic well-being. By merging Spotify-based music genre data with World Happiness Reports (2015-2019), we aim to understand if cultural variety correlates with national prosperity and happiness.

Data Integration & Enrichment
To achieve a high-quality analysis, I integrated two distinct datasets:

Musical Genre Popularity: A dataset containing popularity metrics for six major genre categories across 73 countries.

Socio-Economic Indicators: The World Happiness Report datasets (2015-2019), providing Happiness Scores and GDP per capita metrics.

The datasets were standardized and merged using an inner join on the "Country" variable, resulting in a finalized set of 61 countries for robust statistical analysis.

Original Effort: Shannon Diversity Index
Instead of using raw popularity counts, I implemented the Shannon Diversity Index (H) to quantify "Musical Diversity." This entropy-based metric calculates the balance and variety of music genres within a country:

Formula: H = -Σ (p_i * ln(p_i))

Logic: A higher index indicates a more balanced and diverse musical palette, whereas a lower index indicates a "monoculture" dominated by a single genre. This feature serves as the primary independent variable for my hypothesis testing.

Key EDA Findings
Regional Dominance: Boxplot analysis identified significant outliers in the Latin/Reggaeton genre, indicating high regional concentration compared to the global stability of Pop music.

Cross-Genre Synergy: Correlation heatmaps revealed a strong positive relationship (+0.71) between Hip-hop/R&B and Latin music consumption.

Temporal Stability: Time-series analysis showed that the correlation between music diversity and happiness has remained statistically significant (p < 0.05) and has shown a strengthening trend from 2015 to 2019.

Hypothesis Testing Results
I conducted Pearson Correlation tests to validate the following claims:

Music Diversity vs. Happiness: A moderate positive correlation (r ≈ 0.42) was found with a p-value of 0.0008, leading to the rejection of the Null Hypothesis.

GDP vs. Music Diversity: A positive correlation (r ≈ 0.36) was found with a p-value of 0.0045, suggesting that economic prosperity supports cultural exploration.

Repository Structure
cleaned.csv: Pre-processed music genre dataset.

2015.csv - 2019.csv: Annual happiness report data.

eda_and_testing.py: Comprehensive Python script for data processing, diversity calculation, and statistical testing.

requirements.txt: List of necessary Python libraries (Pandas, NumPy, Scipy, Seaborn, Matplotlib).
