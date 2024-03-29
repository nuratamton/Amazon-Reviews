import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from peft import PeftModel

# Load the fine-tuned model and tokenizer
model_name = "./results/checkpoint-9500"
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=5)

model = PeftModel.from_pretrained("roberta-base", "./results/checkpoint-9500")
model = model.merge_and_unload()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the review data
input_file = "test.csv"
df = pd.read_csv(input_file)
df['Review'] = df['Review'].astype(str)

# Prepare the reviews for the model
def prepare_data(reviews):
    tokenized = tokenizer(reviews, padding=True, truncation=True, max_length=512, return_tensors="pt")
    return tokenized

# Classify reviews and add predictions to the DataFrame
def classify_reviews(df):
    predictions = []
    for review in df['Review']:
        inputs = prepare_data(review).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.append(preds.item()) 
    df['overall'] = predictions 
    return df

# apply classification
df = classify_reviews(df)

output_file = "output_predictions.csv"
df[['id', 'overall']].to_csv(output_file, index=False)
