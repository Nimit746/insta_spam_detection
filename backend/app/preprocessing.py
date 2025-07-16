import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

# Download NLTK resources (only required once)
nltk.download('stopwords')
nltk.download('wordnet')

# ✅ Load the original dataset
df = pd.read_csv(r"C:\Users\sanjeevni\Desktop\insta-spam-detector\backend\data\Final-Dataset.csv")

# ✅ Rename columns for consistency
df.rename(columns={
    'FORMATTED_CONTENT': 'comment',
    'CLASS': 'label'
}, inplace=True)

# ✅ Show column names and label distribution
print("Columns:", df.columns.tolist())
print("\nLabel Distribution:")
print(df['label'].value_counts())

# ✅ Initialize preprocessing tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ✅ Define the cleaning function
def clean_text(text):
    text = str(text).lower()  # Lowercase
    text = re.sub(r'http\S+|www.\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+|\#\w+', '', text)  # Remove mentions and hashtags
    text = text.encode('ascii', 'ignore').decode()  # Remove emojis
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers
    text = text.strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# ✅ Apply text cleaning
df['cleaned_comment'] = df['comment'].apply(clean_text)

# ✅ Show sample cleaned results
print("\nSample Cleaned Data:")
print(df[['comment', 'cleaned_comment']].head())

# ✅ Split (optional, just for sanity check)
X = df['cleaned_comment']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"\nTraining Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")

# ✅ Save cleaned dataset
output_path = r"C:\Users\sanjeevni\Desktop\insta-spam-detector\backend\data\Cleaned-Final-Dataset.csv"
df.to_csv(output_path, index=False)
print(f"\n✅ Cleaned dataset saved to: {output_path}")
