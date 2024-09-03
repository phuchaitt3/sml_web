import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from lime.lime_text import LimeTextExplainer
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the fine-tuned sequence classification model and tokenizer
model_path = './final_fine_tuned_bert_2_class'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading tokenizer and model...")
try:
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)  # Move model to the appropriate device
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")

# Load your DataFrame
df = pd.read_csv('./small_HateXplain_dataset.csv')  # Replace with your actual DataFrame path

# Normalize the input text for matching
df['input_text_normalized'] = df['input_text'].str.lower().str.strip()

# Load the pre-trained Sentence-BERT model
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Compute embeddings for all texts in the dataset
dataset_embeddings = embedding_model.encode(df['input_text'].tolist(), convert_to_tensor=True)

# Fit the LabelEncoder on the labels
label_encoder = LabelEncoder()
label_encoder.fit(df['label'])

# Set the model to evaluation mode
model.eval()

def explain_prediction(text, model, tokenizer, df, label_encoder, dataset_embeddings, device):
    """Generate explanation for a given text using LIME and include the target of hate speech if applicable."""
    model.eval()  
    explainer = LimeTextExplainer(class_names=label_encoder.classes_)

    def model_predict(texts):
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}  
        with torch.no_grad():
            logits = model(**inputs).logits
        return F.softmax(logits, dim=1).cpu().numpy()

    exp = explainer.explain_instance(
        text, 
        model_predict,
        num_features=6,
        top_labels=1  
    )

    predicted_label = exp.top_labels[0]
    predicted_label_name = label_encoder.inverse_transform([predicted_label])[0]

    st.write(f"Predicted Label: {predicted_label_name}")

    if predicted_label_name != 'normal':
        # Compute the embedding for the input text
        input_embedding = embedding_model.encode(text, convert_to_tensor=True)
        
        # Calculate cosine similarity between the input text and all dataset texts
        cosine_similarities = cosine_similarity(input_embedding.cpu().numpy().reshape(1, -1), 
                                                dataset_embeddings.cpu().numpy())[0]

        # Find the index of the most similar text in the dataset
        most_similar_index = np.argmax(cosine_similarities)
        
        # Get the target group for the most similar text
        target_group = df.iloc[most_similar_index]['target']
        st.write(f"Target of Hate Speech: {target_group}")

    exp_html = exp.as_html()
    st.components.v1.html(exp_html, height=800)

# Define Streamlit app
st.title("Hate Speech Classifier with LIME Explanation")

# Get input text from the user
user_input = st.text_area("Enter the text you want to classify:")

if st.button("Classify"):
    explain_prediction(user_input, model, tokenizer, df, label_encoder, dataset_embeddings, device)
else:
    print("Waiting for user input...")
