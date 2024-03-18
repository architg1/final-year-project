import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

csv_file = '/Users/architg/Documents/GitHub/final-year-project/data/wikipedia_corpus_filtered.csv'
df = pd.read_csv(csv_file) # Read the CSV file into a DataFrame

# Specify the output file paths for train, test, and valid splits
train_file_path = '/Users/architg/Documents/GitHub/final-year-project/data/language_model/train.tokens'
test_file_path = '/Users/architg/Documents/GitHub/final-year-project/data/language_model/test.tokens'
valid_file_path = '/Users/architg/Documents/GitHub/final-year-project/data/language_model/valid.tokens'

# Perform train-test-valid split (adjust the test_size and valid_size as needed)
train_data, test_valid_data = train_test_split(df, test_size=0.3, random_state=42)
test_data, valid_data = train_test_split(test_valid_data, test_size=0.5, random_state=42)

# Save each split to the corresponding .tokens file
train_data['unbiased'].to_csv(train_file_path, header=False, index=False)
test_data['unbiased'].to_csv(test_file_path, header=False, index=False)
valid_data['unbiased'].to_csv(valid_file_path, header=False, index=False)

print(f'Train .tokens file created at: {train_file_path}')
print(f'Test .tokens file created at: {test_file_path}')
print(f'Valid .tokens file created at: {valid_file_path}')