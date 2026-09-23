# for data manipulation
import pandas as pd
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("tourism_project/data/tourism.csv")
print("Dataset loaded successfully.")

# Remove unnecassary, CustomerID and unnamed: 0, columns
df.drop(columns=["CustomerID", "Unnamed: 0"], inplace=True)

# Correct inconsistent Gender (Fe male, Female) values
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# Combining Unmarried with Single
df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

# Define the target variable for the classification task
target = "ProdTaken"

# Split independent variables (X) and dependent variable (y)
X = df.drop(columns=[target])
y = df[target]

# stratify=y keeps the (imbalanced) failure ratio consistent across splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save train and test files
X_train.to_csv("Xtrain.csv", index=False)
X_test.to_csv("Xtest.csv", index=False)
y_train.to_csv("ytrain.csv", index=False)
y_test.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
