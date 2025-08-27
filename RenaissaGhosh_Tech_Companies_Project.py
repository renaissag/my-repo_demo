# Set up the data-science environment by importing necessary libraries for tabular analysis, numerical computing, and plotting, ensuring that data 
# loading, wrangling, and visualization can proceed without issues.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Tech_Companies CSV into a pandas DataFrame and display the table to verify that rows and columns have been imported correctly before 
# starting any analysis.
df = pd.read_csv(r'C:\Users\rathi\Tech_Companies_Data.csv')
df

df.dtypes  # Check the data type of each column

# Create a categorical “High_score” flag (threshold: Score ≥ 75) and convert it to readable labels (“Yes”/“No”) to facilitate segmentation and group 
# comparisons during the analysis.
df['High_score'] = df['Score'] >= 75
df['High_score'] = df['High_score'].astype("str")  # Change the data type of the value from boolean to a string
df["High_score"] = df["High_score"].str.replace("False", "No", regex = False)  # Replace "False" with "No" in the "High_score" column
df["High_score"] = df["High_score"].str.replace("True", "Yes", regex = False)  # Replace "True" with "Yes" in the "High_score" column
df

df.dtypes  # Recheck the data type of each column

# Display company-level performance using a diverging bar chart (including a zero line, labels, and title), so stakeholders can easily identify 
# which companies are underperforming and those that are excelling.

# Aggregate performance by company (sum in case of duplicates)
comp_perf = (
    df.groupby("Company")["Performance (in millions)"]
    .sum()
    .reset_index()
    .rename(columns={"Performance (in millions)": "Performance"})
)

# Sort values from lowest to highest
comp_perf = comp_perf.sort_values("Performance", ascending=True).reset_index(drop=True)

# Assign colors: red for negative, green for positive
comp_perf["Color"] = comp_perf["Performance"].apply(lambda x: "red" if x < 0 else "green")

# Plot diverging horizontal bar chart
plt.figure(figsize=(20, 10))
bars = plt.barh(comp_perf["Company"], comp_perf["Performance"], color=comp_perf["Color"])

# Add a vertical line at x=0
plt.axvline(0, color="black", linewidth=1)

# Labels and title
plt.xlabel("Performance (in millions)")
plt.ylabel("Company")
plt.title("Diverging Bar Chart: Company Performance (in millions)")

# Optional grid for readability
plt.grid(axis="x", linestyle="--", alpha=0.5)

# Annotate values on bars
for bar, val in zip(bars, comp_perf["Performance"]):
    x = bar.get_width()
    y = bar.get_y() + bar.get_height() / 2
    align = "left" if x >= 0 else "right"
    offset = 0.02 * (plt.xlim()[1] - plt.xlim()[0])
    plt.text(x + (offset if x >= 0 else -offset), y, f"{val:.2f}",
             va="center", ha=align, fontsize=8)

plt.tight_layout()
plt.show()

# Identify score outliers for each sector using the interquartile range method to find companies that significantly differ from the usual sector 
# performance.

# Create box plots for each sector's score
plt.figure(figsize=(10, 6))  # Creates a new figure with a specified size
sns.boxplot(x="Sector", y="Score", data=df, color="cyan", vert=True)  # Creates a box plot showing the distribution of the score for each sector, using cyan color and vertical orientation
plt.title("Sector-wise Spread of Scores")  # Sets the title of the box plot
plt.xlabel("Sector")  # Sets the x-axis label
plt.ylabel("Score")  # Sets the y-axis label
plt.tight_layout()  # Adjusts the layout of the plot to prevent labels from overlapping
plt.show()  # Displays the box plot

# Identify sectors with outliers (a simple visual inspection from the box plot)
# You'll need to manually look at the box plot to see which sectors have points
# outside the whiskers.

print("\nSectors with potential score outliers (based on visual inspection of the box plots):") # Prints the header indicating departments with outliers
# Based on the generated box plot, list the sectors that show outliers.
# Replace these with the actual sectors you observe with outliers.
sector_outliers = []  # Initializes an empty list to store sectors with outliers
unique_sectors = df["Sector"].unique()  # Gets a list of unique sectors from the DataFrame
for sector in unique_sectors:  # Iterates through each unique department
    sector_data = df[df["Sector"] == sector]["Score"]  # Filters the DataFrame to get score data for the current sector
    q1 = sector_data.quantile(0.25)  # Calculates the first quartile (25th percentile) of the salary data
    q3 = sector_data.quantile(0.75)  # Calculates the third quartile (75th percentile) of the salary data
    iqr = q3 - q1  # Calculates the interquartile range (IQR)
    lower_fence = q1 - 1.5 * iqr  # Calculates the lower fence for outliers
    upper_fence = q3 + 1.5 * iqr  # Calculates the upper fence for outliers
    outliers = sector_data[(sector_data < lower_fence) | (sector_data > upper_fence)]  # Identifies outliers based on the 1.5*IQR rule
    if not outliers.empty:  # Checks if there are any outliers for the current sector
        sector_outliers.append(sector)  # Adds the department name to the list of sectors with outliers

if sector_outliers:  # Checks if there are any departments with outliers
    for sector in sector_outliers:  # Iterates through the list of departments with outliers
        print(f"- {sector}")  # Prints each department with outliers
else:
    print("No sectors with obvious score outliers detected (based on the 1.5*IQR rule).")  # Prints the message if no outliers are found

# Combine and create a comparison table for each sector—Score, Annual Revenue, Performance, Valuation, and Total Funding—to aid in multi-metric 
# benchmarking and reporting.

sector_score = df.groupby("Sector")["Score"].mean()
sector_score_df = pd.DataFrame(sector_score)
sector_revenue = df.groupby("Sector")["Annual Revenue (in millions)"].mean()
sector_revenue_df = pd.DataFrame(sector_revenue)
sector_perf = df.groupby("Sector")["Performance (in millions)"].mean()
sector_perf_df = pd.DataFrame(sector_perf)
sector_val = df.groupby("Sector")["Valuation (in millions)"].mean()
sector_val_df = pd.DataFrame(sector_val)
sector_fund = df.groupby("Sector")["Total Funding (in millions)"].mean()
sector_fund_df = pd.DataFrame(sector_fund)
sector_comb_table = pd.concat([sector_score_df, sector_revenue_df, sector_perf_df, sector_val_df, sector_fund_df], axis = 1)
sector_comb_table

sector_comb_table.columns  # Display the column names of this table

# Create a comparison dashboard with five box plots (Score, Annual Revenue, Performance, Valuation, and Total Funding for all sectors) to illustrate 
# the distribution, range, and possible outliers for each metric.

# Dashboard containing 5 box plots for each subject
fig, axis = plt.subplots(1, 5, figsize=(16,6), sharex=True, sharey=True)
mydata = [sector_comb_table["Score"], sector_comb_table["Annual Revenue (in millions)"], sector_comb_table["Performance (in millions)"], \
         sector_comb_table["Valuation (in millions)"], sector_comb_table["Total Funding (in millions)"]]
myheadings = ['Score', 'Annual Revenue (in millions)', 'Performance (in millions)', \
       'Valuation (in millions)', 'Total Funding (in millions)']
mycolors = ["yellow", "cyan", "pink", "lavender", "bisque"]

for axis, mydata, myheadings, mycolors in zip(axis, mydata, myheadings, mycolors):
    axis.boxplot(mydata, vert=True)
    axis.set_title(myheadings)
    axis.set_facecolor(mycolors)
plt.tight_layout()
plt.show()

# Calculate essential summary statistics (mean, median, mode, variance, standard deviation, coefficient of variation for all sectors) to describe the 
# central tendency and spread of key numeric fields, and produce a clear summary table.

# Calculating Central Tendencies (measures of mean, median, and mode), variance, standard deviation, and coefficient of variation
mean = []
median = []
mode = []
variance = []
std_dev = []
coefficient = []

for col in sector_comb_table.columns:
    mean.append(sector_comb_table.loc[:,col].mean())
    median.append(sector_comb_table.loc[:,col].median())
    mode.append(sector_comb_table.loc[:,col].mode()[0])

    variance.append(sector_comb_table.loc[:,col].var())
    mean_val = sector_comb_table.loc[:,col].mean()
    stddev = sector_comb_table.loc[:,col].std()
    std_dev.append(sector_comb_table.loc[:,col].std())
    coefficient.append(stddev / mean_val)

sector_new_df = pd.DataFrame([mean, median, mode, variance, std_dev, coefficient],
                       columns=['Score', 'Annual Revenue (in millions)', 'Performance (in millions)', \
       'Valuation (in millions)', 'Total Funding (in millions)'],
                       index = ["Mean", "Median", "Mode", "Variance", "Standard Deviation", "Coefficient of Variation"])
sector_new_df

# Combine and create a comparison table for each company—Score, Annual Revenue, Performance, Valuation, and Total Funding—to aid in multi-metric 
# benchmarking and reporting.

comp_score = df.groupby("Company")["Score"].sum()
comp_score_df = pd.DataFrame(comp_score)
comp_revenue = df.groupby("Company")["Annual Revenue (in millions)"].sum()
comp_revenue_df = pd.DataFrame(comp_revenue)
comp_perf = df.groupby("Company")["Performance (in millions)"].sum()
comp_perf_df = pd.DataFrame(comp_perf)
comp_val = df.groupby("Company")["Valuation (in millions)"].sum()
comp_val_df = pd.DataFrame(comp_val)
comp_fund = df.groupby("Company")["Total Funding (in millions)"].sum()
comp_fund_df = pd.DataFrame(comp_fund)
comp_comb_table = pd.concat([comp_score_df, comp_revenue_df, comp_perf_df, comp_val_df, comp_fund_df], axis = 1)
comp_comb_table

comp_comb_table.columns  # Display the column names of this table

# Create a comparison dashboard with five box plots (Score, Annual Revenue, Performance, Valuation, and Total Funding for all companies) to illustrate 
# the distribution, range, and possible outliers for each metric.

# Dashboard containing 5 box plots for each subject
fig, axis = plt.subplots(1, 5, figsize=(16,6), sharex=True, sharey=True)
mydata = [comp_comb_table["Score"], comp_comb_table["Annual Revenue (in millions)"], comp_comb_table["Performance (in millions)"], \
         comp_comb_table["Valuation (in millions)"], comp_comb_table["Total Funding (in millions)"]]
myheadings = ['Score', 'Annual Revenue (in millions)', 'Performance (in millions)', \
       'Valuation (in millions)', 'Total Funding (in millions)']
mycolors = ["yellow", "cyan", "pink", "lavender", "bisque"]

for axis, mydata, myheadings, mycolors in zip(axis, mydata, myheadings, mycolors):
    axis.boxplot(mydata, vert=True)
    axis.set_title(myheadings)
    axis.set_facecolor(mycolors)
plt.tight_layout()
plt.show()

# Calculate essential summary statistics (mean, median, mode, variance, standard deviation, coefficient of variation for all sectors) to describe the 
# central tendency and spread of key numeric fields, and produce a clear summary table.

# Calculating Central Tendencies (measures of mean, median, and mode), variance, standard deviation, and coefficient of variation
mean = []
median = []
mode = []
variance = []
std_dev = []
coefficient = []

for col in comp_comb_table.columns:
    mean.append(comp_comb_table.loc[:,col].mean())
    median.append(comp_comb_table.loc[:,col].median())
    mode.append(comp_comb_table.loc[:,col].mode()[0])

    variance.append(comp_comb_table.loc[:,col].var())
    mean_val = comp_comb_table.loc[:,col].mean()
    stddev = comp_comb_table.loc[:,col].std()
    std_dev.append(comp_comb_table.loc[:,col].std())
    coefficient.append(stddev / mean_val)

comp_new_df = pd.DataFrame([mean, median, mode, variance, std_dev, coefficient],
                       columns=['Score', 'Annual Revenue (in millions)', 'Performance (in millions)', \
       'Valuation (in millions)', 'Total Funding (in millions)'],
                       index = ["Mean", "Median", "Mode", "Variance", "Standard Deviation", "Coefficient of Variation"])
comp_new_df

# Investigate how the distribution of "High_score" changes across important categorical features (Sector, Founding Year, City, State) using count plots, 
# to reveal where top performers are concentrated.

# Create a count plot for each category-wise high-score count
df['Founding Year'] = df['Founding Year'].astype("str")
categ = df.select_dtypes(exclude="number")
high_score_col = categ["High_score"]

cols = ["Sector", "Founding Year", "City", "State"]
def mycountplot(cols):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=categ[cols], hue=high_score_col)
    plt.title(f"Company High-Score Count by {cols}")
    plt.xlabel(cols)
    plt.ylabel("Company High-Score Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

mycountplot("Sector")
mycountplot("Founding Year")
mycountplot("City")
mycountplot("State")

# Generate two pie charts to visualize the sector-wise distribution of companies with high scores (≥75) and low scores (<75).

# Create a pie chart for sector-wise high-score companies (High_score = 'Yes')
high_score_col = categ["High_score"]
sector_col = df["Sector"]

sector_df = pd.DataFrame({"Sector": sector_col, "High Score": high_score_col})
yes_by_sector = sector_df[sector_df["High Score"] == "Yes"]["Sector"].value_counts()

labels = yes_by_sector.index
sizes = yes_by_sector.values

plt.figure(figsize=(10, 6))
plt.pie(sizes, labels=labels, autopct='%.2f %%')
plt.title("Ratio of High-Score Companies by Sector")
plt.tight_layout()
plt.show()

# Create a pie chart for sector-wise low-score companies (High_score = 'No')
no_by_sector = sector_df[sector_df["High Score"] == "No"]["Sector"].value_counts()
labels = no_by_sector.index
sizes = no_by_sector.values

plt.figure(figsize=(10, 6))
plt.pie(sizes, labels=labels, autopct='%.2f %%')
plt.title("Ratio of Low-Score Companies by Sector")
plt.tight_layout()
plt.show()

# Visualize the distribution of each numerical variable exclusively for only high-scoring companies.

# Create a histogram for each of the numerical columns with the 'High_score' column
conti = df.select_dtypes(include="number")
high_score_col = categ["High_score"]
high_score_label = "Yes"
for i in conti.columns:
    plt.figure(figsize=(10, 6))
    hist = conti[i][high_score_col == high_score_label].dropna()
    n, bins, patches = plt.hist(hist, edgecolor="black", label=f"High Score: {high_score_label}")
    plt.xticks(bins)
    plt.title(f"Distribution of {i}")
    plt.xlabel(i)
    plt.ylabel(f"Company High-Score Count")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Compare the distribution and variability of numerical metrics between high-scoring and low-scoring companies using boxplots.

# Create a boxplot for each of the numerical columns with the 'High_score' column
for i in conti.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=conti[i], y=categ["High_score"])
    plt.tight_layout()
    plt.show()

# Display the rise of high-performing companies only in Fintech, Enterprise Tech, Defence Tech, and Space Tech across their founding years, with a 
# horizontal bar chart.

# Create a horizontal bar chart for the year-wise high-scorers from the Fintech, Enterprise Tech, Defence Tech, and Space Tech sectors
filtered_df = df[df["Sector"].isin(["Fintech", "Enterprise Tech", "Defence Tech", "Space Tech"]) & (df["High_score"] == "Yes")]
yearly_high_score = filtered_df["Founding Year"].value_counts().sort_values()
plt.figure(figsize=(10, 6))
plt.barh(yearly_high_score.index, yearly_high_score.values, color="cyan")
plt.xlabel("Number of High-Scorers")
plt.ylabel("Year")
plt.title("Year-wise High-Scorers from Fintech, Enterprise Tech, Defence Tech, and Space Tech Sectors")
plt.tight_layout()
plt.show()

# Determine the total mean of yearly revenue in all companies.
avg_rev = df["Annual Revenue (in millions)"].mean()
print(avg_rev)

# Determine the total mean of yearly revenue and compare it with the mean values in high-score and low-score companies.

data_filter_1 = df[df["High_score"] == "Yes"]
avg_1 = data_filter_1["Annual Revenue (in millions)"].mean()
print(avg_1)

data_filter_2 = df[df["High_score"] == "No"]
avg_2 = data_filter_2["Annual Revenue (in millions)"].mean()
print(avg_2)

# Select a simple random sample of 16 companies with replacement, then calculate and compare the average revenues for all companies, high scorers, 
# and low scorers.

# Simple Random Sampling

rand_sample = df.sample(n=16, random_state=42, replace=True)

avg_rand_sample = rand_sample["Annual Revenue (in millions)"].mean()
print(avg_rand_sample)

data_rand_sample_1 = rand_sample[rand_sample["High_score"] == "Yes"]
avg_rand_sample_1 = data_rand_sample_1["Annual Revenue (in millions)"].mean()
print(avg_rand_sample_1)

data_rand_sample_2 = rand_sample[rand_sample["High_score"] == "No"]
avg_rand_sample_2 = data_rand_sample_2["Annual Revenue (in millions)"].mean()
print(avg_rand_sample_2)

# Conduct systematic sampling by choosing every k-th company (adjusted to get 16 samples). Then, calculate the average revenues for the overall sample 
# and for high-score and low-score groups to assess how effective systematic selection is in approximating population metrics.

# Systematic Sampling

n = 16  # sample size
my_rows = len(df) // 16  # Adjusting for 16 samples
systematic_sample = df.iloc[::my_rows].head(16)

avg_sys_sample = systematic_sample["Annual Revenue (in millions)"].mean()
print(avg_sys_sample)

data_sys_sample_1 = systematic_sample[systematic_sample["High_score"] == "Yes"]
avg_sys_sample_1 = data_sys_sample_1["Annual Revenue (in millions)"].mean()
print(avg_rand_sample_1)

data_sys_sample_2 = systematic_sample[systematic_sample["High_score"] == "No"]
avg_sys_sample_2 = data_rand_sample_2["Annual Revenue (in millions)"].mean()
print(avg_sys_sample_2)

# Use stratified sampling by splitting the dataset into groups based on scores and taking 50% from each group. Compare the average revenues of the 
# combined stratified sample and subgroups (high-score vs. low-score) to evaluate the method’s representativeness.

# Stratified Sampling

samples = []
grouped_df = df.groupby("Score")
for group_name, group_data in grouped_df:
    sample_data = group_data.sample(frac=0.5, random_state=42)
    samples.append(sample_data)
strat_sample_df = pd.concat(samples, ignore_index=True)

avg_strat_sample = strat_sample_df["Annual Revenue (in millions)"].mean()
print(avg_strat_sample)

data_strat_sample_1 = strat_sample_df[strat_sample_df["High_score"] == "Yes"]
avg_strat_sample_1 = data_strat_sample_1["Annual Revenue (in millions)"].mean()
print(avg_strat_sample_1)

data_strat_sample_2 = strat_sample_df[strat_sample_df["High_score"] == "No"]
avg_strat_sample_2 = data_strat_sample_2["Annual Revenue (in millions)"].mean()
print(avg_strat_sample_2)

# Group companies into two clusters based on the “High_score” variable, then take a 50% sample from each cluster. Then, calculate the average annual 
# revenues for the overall cluster sample and subgroups to analyze the effectiveness of cluster sampling for subgroup analysis.

# Cluster Sampling

prem_group = df.groupby("High_score")  # Grouping by 'premium'
clusters = ["Yes", "No"]
my_list = []  # Make sure this matches
for i in clusters:
    cluster_data = prem_group.get_group(i)
    sample_data = cluster_data.sample(frac=0.5, random_state=4)
    my_list.append(sample_data)  # Use correct variable name
cluster_sample_df = pd.concat(my_list, ignore_index=True)  # Concatenate the sampled clusters

# Overall cluster sample average price
avg_cluster_sample = cluster_sample_df["Annual Revenue (in millions)"].mean()
print(avg_cluster_sample)

# Premium and Non-Premium average prices in a cluster sample
data_cluster_sample_1 = cluster_sample_df[cluster_sample_df["High_score"] == "Yes"]
avg_cluster_sample_1 = data_cluster_sample_1["Annual Revenue (in millions)"].mean()
print(avg_cluster_sample_1)

data_cluster_sample_2 = cluster_sample_df[cluster_sample_df["High_score"] == "No"]
avg_cluster_sample_2 = data_cluster_sample_2["Annual Revenue (in millions)"].mean()
print(avg_cluster_sample_2)

# Interpretations:
# 1. Sample means vary slightly depending on the technique used. Cluster sampling comes closest to the population average, which is expected as it maintains the high-score/low-score company ratio.
# 2. In the full dataset, low-score companies surprisingly have a higher average annual revenue than the high-score ones.
# 3. In the sampled data, this pattern remains, though both categories appear slightly more expensive in the sample.
# 4. This indicates sampling variation, but overall, low-scoring companies are more expensive on average, which might be due to fewer non-high-end units being high-end or having exceptional specs driving up their average.
# 5. Cluster sampling provided a closer estimate to the real population average, as it preserved the balance between high-score and low-score companies.

# Create samples of different sizes and plot their distributions. Compare sample means to show how larger sample sizes decrease variability and lead 
# to convergence toward the population mean, illustrating the Central Limit Theorem. 

# Draw distribution plots to test for different sample sizes

num = [5, 10, 15, 20, 25, 30] 
data_s = []
data_s_mean = []
sample_df = pd.DataFrame()

fig, ax = plt.subplots(2, 4, figsize=(20, 10))
for i in num:
    sample_df=df.sample(n=i, random_state=1, replace=True) 
    data_s.append(sample_df["Annual Revenue (in millions)"].tolist())
    data_s_mean.append(sample_df["Annual Revenue (in millions)"].mean())
    
k=0
for i in range(0, 2):
    for j in range(0, 3):
        sns.distplot(data_s[k],ax=ax[i, j])
        ax[i, j].set_title(label = 'Sample size='+str(len(data_s[k])))
        k = k + 1
plt.show()
print(data_s_mean)

# Interpretations:
# 1. As the sample size increases, the variability of sample means decreases.
# 2. The distribution of sample means becomes more normal as sample size increases, even if the original population is right-skewed.
# 3. The sample means center around the population mean, and their spread approximates σ/sqrt(n), satisfying the CLT.

# Import necessary libraries for statistical analysis, computing, and plotting, ensuring that data loading, wrangling, and visualization can proceed 
# without issues.
import statistics as st
import scipy.stats as sts
from scipy.stats import norm
from scipy.stats import t

# Compute descriptive statistics (mean, median, standard deviation, skewness, kurtosis) for yearly revenue. Use these metrics to create a fitted normal 
# distribution curve, examining how well company revenue aligns with a normal distribution and spotting any deviations from normality.

data_p = df["Annual Revenue (in millions)"].tolist()
p_mean = st.mean(data_p)
p_median = st.median(data_p)
p_sd = st.stdev(data_p)
print(p_mean)
print(p_median)
print(p_sd)
print(sts.skew(data_p, bias = False))
print(sts.kurtosis(data_p, bias = False))
# If we write 'bias = False', then we are correcting bias.
# If we write 'bias = True', then we are not correcting bias.
lower_p = p_mean - (4*p_sd)
upper_p = p_mean + (4*p_sd)
norm_p = np.arange(lower_p, upper_p)
print(norm_p)
plt.title("Normal Distribution Curve")
plt.plot(norm_p, norm.pdf(norm_p, p_mean, p_sd));

# Conduct a test to assess if the average annual revenue for 2023 exceeds the 2022 benchmark of 22.2. State the null and alternative hypotheses, 
# use a sample size of 32, and compute the critical value, statistic, and p-value at a 5% significance level. Decide if the null hypothesis should 
# be rejected and provide a business conclusion on whether the company can claim that 2023 revenues exceed those of 2022, supported by a plotted 
# population distribution curve for visualization.

# State the null and alternate hypotheses
# The analysis refers to the population, not the sample. So, we use the population mean (mu).
# H_0: mu <= 22.2, where n = 32.
# H_a: mu > 22.2, where n = 32.
# Sample size (n = 32) is greater than 30, so we use z-test.
# One-sampled one-sided right-tailed z-test is to be performed.

# Calculate the z-critical value
t_alpha = 0.05  #Alpha is 0.05 since it is a one-tailed z-test.
t_z_critical = norm.ppf(1 - t_alpha)
print(t_z_critical)

# A one-sample z-test is used for the hypothesis test.
t_pop_mean = 22.2
t_samp_mean = 40
t_pop_std = 30.5
t_samp_size = 32
t_z_stats = (t_samp_mean - t_pop_mean) / (t_pop_std / t_samp_size**0.5)  # Calculating the z-statistic value
print(t_z_stats)

# Calculate the p-value
t_p_value = norm.sf(abs(t_z_stats))
print(t_p_value)

# Conclude to determine whether we accept or reject the null hypothesis (Check the conditions)
print(t_p_value < t_alpha)
print(t_z_stats > t_z_critical)
print("Conclusion: Since p-value < alpha and z-statistic > z-critical in a one-tailed z-test, we reject the null hypothesis.")
print("Business Conclusion: The statistical analysis indicates that the 2023 annual revenue is not significantly higher than the 2022 revenue, so the company should not assume an increase.")

# Range calculation
t_lower_val = t_pop_mean - (4 * t_pop_std)
t_upper_val = t_pop_mean + (4 * t_pop_std)
t_x = np.arange(t_lower_val, t_upper_val)  # Calculating the range

# Plot the distribution curve
print(t_x)
plt.title("Population Distribution Curve")
plt.plot(t_x, norm.pdf(t_x, t_pop_mean, t_pop_std));

# Conduct a test to determine if there is a statistically significant difference in mean annual revenue between high-score and low-score companies. 
# State the null and alternative hypotheses, and calculate the critical value at a 5% significance level, the test statistic, and the p-value. Decide 
# whether to reject or accept the null hypothesis, followed by a business interpretation of the revenue differences between the two groups. Plot the 
# distribution curves of annual revenues for high-score and low-score companies to visually compare their distributions.

# State the null and alternate hypotheses
# The analysis refers to the population, not the sample. So, we use the population mean (mu).
# H_0: mu1 = mu2
# H_a: mu1 != mu2
# Since the sample sizes (n1 = 21 and n2 = 10) are less than 30, a t-test is used.
# 2-sampled two-tailed t-test is to be performed.

# Find the number of low-score and high-score companies
alpha = 0.05
data_filter_1 = df[df["High_score"] == "Yes"]
data_filter_2 = df[df["High_score"] == "No"]
n1 = len(data_filter_1)
print(n1)
n2 = len(data_filter_2)
print(n2)

# Calculate the critical value
dof = n1 + n2 - 2  # Calculating degrees of freedom
print(dof)
alpha2 = 0.05
t_critical = t.ppf((1 - alpha2/2), dof)  # Calculating the t-critical value - Since it is a two-tailed t-test, alpha ('alpha2') will be divided by 2; Positive t-critical value → upper bound for rejection region.
print(t_critical)

# Calculate the test statistic value and p-value
x1 = st.mean(data_filter_1["Annual Revenue (in millions)"])  # Calculating the sample mean of annual revenue of high-score companies
x2 = st.mean(data_filter_2["Annual Revenue (in millions)"])  # Calculating the sample mean of annual revenue of low-score companies
v1 = (st.stdev(data_filter_1["Annual Revenue (in millions)"])**2)  # Calculating the sample variance of annual revenue of high-score companies
v2 = (st.stdev(data_filter_2["Annual Revenue (in millions)"])**2)  # Calculating the sample variance of annual revenue of low-score companies
signal = abs(x1 - x2)  # Calculating the signal
noise = (v1 * (n1 - 1) + v2 * (n2 - 1))**0.5 * ((1 / n1) + (1 / n2))**0.5  # Calculating the noise
t_statistic = signal / noise  # Calculating the test (t-statistic) statistic value
print(t_statistic)  # Prints the test (t-statistic) statistic value
p_value = t.sf(abs(t_statistic), dof)*2  # Calculating the p-value - Since it is a two-tailed t-test, p-value will be multiplied by 2.
print(p_value)  # Prints the p-value

# Conclude in determining whether we accept or reject the null hypothesis (Check the conditions)
print(p_value > alpha2)
print(t_statistic > t_critical)
print(-t_critical > t_statistic)
print("Conclusion: Since p-value > alpha and t_statistic > t_critical in a two-tailed t-test, we accept the null hypothesis.")
print("Business Conclusion: There is a statistically significant difference in the mean annual revenue of high-score and low-score companies.")

# Plot the distribution curve
lower_val_1 = x1 - (4 * st.stdev(data_filter_1["Annual Revenue (in millions)"]))  # Calculating the lower value of annual revenues of high-score companies
lower_val_2 = x2 - (4 * st.stdev(data_filter_2["Annual Revenue (in millions)"]))  # Calculating the upper value of annual revenues of low-score companies
upper_val_1 = x1 + (4 * st.stdev(data_filter_1["Annual Revenue (in millions)"]))  # Calculating the lower value of annual revenues of high-score companies
upper_val_2 = x2 + (4 * st.stdev(data_filter_2["Annual Revenue (in millions)"]))  # Calculating the upper value of annual revenues of low-score companies
t_x1 = np.arange(lower_val_1, upper_val_1)  # Calculating the range of annual revenues of high-score companies
t_x2 = np.arange(lower_val_2, upper_val_2)  # Calculating the range of annual revenues of low-score companies

print(t_x1)  # Prints the range of annual revenues of high-score companies
print(t_x2)  # Prints the range of annual revenues of low-score companies
plt.title("Sample Distribution Curve")  # Adds the title of the distribution plot
t1 = plt.plot(t_x1, norm.pdf(t_x1, x1, st.stdev(data_filter_1["Annual Revenue (in millions)"])), label="Annual Reveunue of High-Score Companies (in millions)");  # Plot the distribution curve, and add a label to create a legend for annual revenues of high-score companies
t2 = plt.plot(t_x2, norm.pdf(t_x2, x2, st.stdev(data_filter_2["Annual Revenue (in millions)"])), label="Annual Reveunue of Low-Score Companies (in millions)");  # Plot the distribution curve, and add a label to create a legend for annual revenues of low-score companies
plt.legend()  # Creates a legend for the distribution curves for annual revenues of high-score and low-score companies
plt.show()  # Displays the plot of the distribution curves
