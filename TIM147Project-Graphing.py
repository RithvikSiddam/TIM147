import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

print("New run!")

usecols = ['DISCOVERY_DOY',
    'NWCG_GENERAL_CAUSE',
    'CONT_DOY',
    'FIRE_SIZE',
    'FIRE_SIZE_CLASS',
    'LATITUDE',
    'LONGITUDE',
    'STATE',
    'COUNTY',
    'NPL',
    'Mang_Type',
    'Des_Tp',
    'GAP_Sts',
    'GAP_Prity',
    'EVH',
    'EVT',
    'EVC',
    'Land_Cover',
    'rpms',
    'Population',
    'GACCAbbrev',
    'GACC_PL',
    'GACC_New fire',
    'GACC_Type 1 IMTs',
    'GACC_Type 2 IMTs',
    'GACC_NIMO Teams',
    'GACC_Area Command Teams',
    'GACC_Fire Use Teams',
    'pr_Normal',
    'tmmn_Normal',
    'tmmx_Normal',
    'rmin_Normal',
    'rmax_Normal',
    'sph_Normal',
    'srad_Normal',
    'fm1000',
    'bi_Normal',
    'vpd_Normal',
    'erc_Normal',
    'DF_PFS',
    'AF_PFS',
    'M_WTR',
    'M_CLT',
    'No_FireStation_20.0km',
    'FRG',
    'TRI_1km',
    'TRI',
    'Aspect_1km',
    'Aspect',
    'Elevation_1km',
    'Elevation',
    'Slope_1km',
    'Slope',
    'Annual_etr',
    'Annual_precipitation',
    'Annual_tempreture',
    'Aridity_index',
    'pr',
    'tmmn',
    'tmmx',
    'rmin',
    'rmax',
    'sph',
    'vs',
    'th',
    'srad',
    'etr',
    'fm100',
    'fm1000',
    'bi',
    'vpd',
    'erc',
    'pr_5D_mean',
    'tmmn_5D_mean',
    'tmmx_5D_mean',
    'rmin_5D_mean',
    'rmax_5D_mean',
    'sph_5D_mean',
    'vs_5D_mean',
    'th_5D_mean',
    'srad_5D_mean',
    'etr_5D_mean',
    'fm100_5D_mean',
    'fm1000_5D_mean',
    'bi_5D_mean',
    'vpd_5D_mean',
    'erc_5D_mean',
    'NDVI-1day']

print("Listed cols")

# Define converters to strip whitespace and handle bad values
def clean_float(x):
    try:
        x = x.strip() if isinstance(x, str) else x
        return float(x)
    except:
        return np.nan

def clean_int(x):
    try:
        x = x.strip() if isinstance(x, str) else x
        return int(float(x))  # handles '3.0' as well
    except:
        return pd.NA

# Column-specific converters

converters = {
    'DISCOVERY_DOY': clean_int,
    'CONT_DOY': clean_float,
    'FIRE_SIZE': clean_float,
    'LATITUDE': clean_float,
    'LONGITUDE': clean_float,
    'EVH': clean_float,
    'EVT': clean_float,
    'EVC': clean_float,
    'Land_Cover': clean_float,
    'rpms': clean_int,
    'Population': clean_float,
    'GACC_New fire': clean_float,
    'GACC_Type 1 IMTs': clean_float,
    'GACC_Type 2 IMTs': clean_float,
    'GACC_NIMO Teams': clean_float,
    'GACC_Area Command Teams': clean_float,
    'GACC_Fire Use Teams': clean_float,
    'pr_Normal': clean_float,
    'tmmn_Normal': clean_float,
    'tmmx_Normal': clean_float,
    'rmin_Normal': clean_float,
    'rmax_Normal': clean_float,
    'sph_Normal': clean_float,
    'srad_Normal': clean_float,
    'fm1000': clean_float,
    'bi_Normal': clean_float,
    'vpd_Normal': clean_float,
    'erc_Normal': clean_float,
    'DF_PFS': clean_float,
    'AF_PFS': clean_float,
    'M_WTR': clean_float,
    'M_CLT': clean_float,
    'No_FireStation_20.0km': clean_float,
    'TRI_1km': clean_float,
    'TRI': clean_float,
    'Aspect_1km': clean_float,
    'Aspect': clean_float,
    'Elevation_1km': clean_float,
    'Elevation': clean_int,
    'Slope_1km': clean_float,
    'Slope': clean_int,
    'Annual_etr': clean_int,
    'Annual_precipitation': clean_int,
    'Annual_tempreture': clean_float,
    'Aridity_index': clean_float,
    'pr': clean_float,
    'tmmn': clean_float,
    'tmmx': clean_float,
    'rmin': clean_float,
    'rmax': clean_float,
    'sph': clean_float,
    'vs': clean_float,
    'th': clean_float,
    'srad': clean_float,
    'etr': clean_float,
    'fm100': clean_float,
    'fm1000': clean_float,
    'bi': clean_float,
    'vpd': clean_float,
    'erc': clean_float,
    'pr_5D_mean': clean_float,
    'tmmn_5D_mean': clean_float,
    'tmmx_5D_mean': clean_float,
    'rmin_5D_mean': clean_float,
    'rmax_5D_mean': clean_float,
    'sph_5D_mean': clean_float,
    'vs_5D_mean': clean_float,
    'th_5D_mean': clean_float,
    'srad_5D_mean': clean_float,
    'etr_5D_mean': clean_float,
    'fm100_5D_mean': clean_float,
    'fm1000_5D_mean': clean_float,
    'bi_5D_mean': clean_float,
    'vpd_5D_mean': clean_float,
    'erc_5D_mean': clean_float,
    'NDVI-1day': clean_float,
    'CheatGrass': clean_float,
    'ExoticAnnualGrass': clean_float,
    'Medusahead': clean_float,
    'PoaSecunda': clean_float
}
print("Converted cols")

dtype = {
'NWCG_GENERAL_CAUSE': 'object',
    'FIRE_SIZE_CLASS': 'object',
    'STATE': 'object',
    'COUNTY': 'object',
    'NPL': 'object',
    'Mang_Type': 'object',
    'Des_Tp': 'object',
    'GAP_Sts': 'object',
    'GAP_Prity': 'object',
    'GACCAbbrev': 'object',
    'GACC_PL': 'object',
    'FRG': 'object'
}
print("Typed objects")

# Load cleaned data
df = pd.read_csv(
    '/Users/rithviks/Desktop/TIM147/FPA_FOD_Plus.csv',
    usecols=usecols,
    converters=converters,
    dtype=dtype,
    low_memory=False
)

print("Loaded data successfully.")
print(df.dtypes)

# Assuming df is loaded successfully

# Categorical columns: these are columns that are objects or categories
categorical_cols = df.select_dtypes(include=['object']).columns

# Numerical columns: these are columns that are float or int
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Graphing Categorical Variables
for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=col, palette='Set2')
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=90)  # Rotate x-axis labels for readability if necessary
    plt.tight_layout()
    plt.show()

# Graphing Numerical Variables
for col in numerical_cols:
    plt.figure(figsize=(10, 6))
    
    # Plot a histogram for the numerical variables
    sns.histplot(df[col], kde=True, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()

    # Optional: You can also create box plots for numerical variables to show outliers
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[col], color='lightcoral')
    plt.title(f'Box plot of {col}')
    plt.tight_layout()
    plt.show()

    # Optional: If you want to explore relationships between numerical variables, you can create scatter plots
    # For example, scatter plot of 'FIRE_SIZE' vs. 'LATITUDE'
    if 'FIRE_SIZE' in numerical_cols and 'LATITUDE' in numerical_cols:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='LATITUDE', y='FIRE_SIZE', color='green')
        plt.title(f'Scatter plot of FIRE_SIZE vs LATITUDE')
        plt.tight_layout()
        plt.show()
