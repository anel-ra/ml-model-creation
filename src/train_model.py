import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv("data/products.csv")

# drop all rows with missing values
df = df.dropna()
df.columns = df.columns.str.strip()

# Convert all sentiment values to lowercase and strip extra spaces
df['Category Label'] = df['Category Label'].astype(str).str.lower().str.strip()

# Convert column type to 'category'
df['Category Label'] = df['Category Label'].astype('category')

# Drop columns that are not useful for modeling
df = df.drop(columns=['product ID', 'Merchant ID', '_Product Code', 'Number_of_Views', 'Merchant Rating', 'Listing Date'])

# Create new column with length of each review_text
df['Title Length'] = df['Product Title'].astype(str).str.len()

# Define features and label
X = df[["Product Title", "Title Length"]]
y = df["Category Label"]

# Define preprocessing
preprocessor = ColumnTransformer(
    transformers=[    
        ("title", TfidfVectorizer(), "Product Title"),
        ("length", MinMaxScaler(), ["Title Length"])
    ]
)

# Define pipeline with the best model (e.g. RandomForestClassifier)
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression())
])

# Train the model on the entire dataset
pipeline.fit(X, y)

# Save the model to a file
joblib.dump(pipeline, "model/predict_category_model.pkl")

print(" Model trained and saved as 'model/predict_category_model.pkl'")
