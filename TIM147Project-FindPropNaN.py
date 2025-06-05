import pandas as pd

# Load your data (update the path to your file)
df = pd.read_csv('/Users/rithviks/Desktop/TIM147/FPA_FOD_Cleaned.csv')
"""
# Count frequency of each COUNTY value including NaNs
county_counts = df["COUNTY"].value_counts(dropna=False)

# Calculate proportions
county_props = county_counts / len(df)

# Print COUNTY values and their proportions
print(county_props.sort_values(ascending=False))
"""
#df["COUNTY"] = df["COUNTY"].fillna("Missing")

# Calculate proportion of missing values per column
missing_per_column = df.isnull().mean()

print("Proportion of missing values per column:")
print(missing_per_column)

# Calculate overall proportion of missing values in the entire DataFrame
total_missing = df.isnull().sum().sum()
total_values = df.size
overall_missing_proportion = total_missing / total_values

print(f"\nOverall proportion of missing values in the dataset: {overall_missing_proportion:.4f}")

#fill_zero_columns = [
#    "CheatGrass", "ExoticAnnualGrass", "Medusahead", "PoaSecunda"
#]
#df[fill_zero_columns] = df[fill_zero_columns].fillna(0)
#df = df.drop(columns=["CONT_TIME"])
#df = df.drop(columns=["CONT_DOY"])

# Shows that GACC_PL(which is missing 51% of values) has <0.01 correlation with target vars. Dropping.
#target_cols = ["EALR_PFS", "EBLR_PFS", "EPLR_PFS"]
#for target in target_cols:
#    corr = df["GACC_PL"].corr(df[target])
#    print(f"Correlation between GACC_PL and {target}: {corr:.4f}")

#df = df.drop(columns=["GACC_PL"])
# Define your target columns
#target_cols = ["EALR_PFS", "EBLR_PFS", "EPLR_PFS"]

# Select only numeric features (correlation only works with numeric types)
#numeric_features = df.select_dtypes(include=["number"]).drop(columns=target_cols)

# Loop through each target and calculate correlation with all features
#for target in target_cols:
#    print(f"\n--- Correlation with {target} ---")
#    correlations = numeric_features.corrwith(df[target])
#    sorted_corr = correlations.abs().sort_values(ascending=False)
#    print(sorted_corr[sorted_corr > 0.1])  # Show only "potentially useful" ones

#df = df.drop(columns=["Population"])
#df = df.drop(columns=["COUNTY"])
#df = df.drop(columns=["DISCOVERY_TIME"])

#df = df.drop(columns=["GHM"])
#df = df.drop(columns=["TPI_1km"])
#df = df.drop(columns=["RPL_THEMES"])

#df = df.dropna()

#df.to_csv("/Users/rithviks/Desktop/TIM147/FPA_FOD_Cleaned.csv", index=False)
