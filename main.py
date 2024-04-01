'''
    Program Developed by: Kristian Vazquez

'''

# Importing necessary libraries
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

# Load the dataset
crohn_patients = pd.read_csv('./data/crohn_patients.csv')

# Making sure the data is printed correctly
print(crohn_patients)
print(crohn_patients.columns)

# Split the "Preferred Foods" column into lists of foods
crohn_patients['Preferred Foods'] = crohn_patients['Preferred Foods'].str.split(', ')

# Apply one-hot encoding to categorical variables
encoded_df = pd.get_dummies(crohn_patients, columns=['Gender', "Crohn's Disease Severity", 'Medication'])

# Split the "Preferred Foods" column into lists of foods
encoded_df['Preferred Foods'] = encoded_df['Preferred Foods'].str.split(', ')

# Apply one-hot encoding to the "Preferred Foods" column
encoded_df = encoded_df.explode('Preferred Foods')
encoded_df = pd.get_dummies(encoded_df, columns=['Preferred Foods'])

# Drop unnecessary columns
encoded_df.drop(columns=['Patient ID', 'Age'], inplace=True)

# Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(encoded_df, min_support=0.1, use_colnames=True)

# Generate assocaition rules
association_rules_df = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)

# Filter the rules based on confidence and support
filtered_rules = association_rules_df[(association_rules_df['confidence'] > 0.6) & (association_rules_df['support'] > 0.1)]

# Print the filtered rules
print(filtered_rules)