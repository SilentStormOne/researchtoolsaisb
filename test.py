import json
import pandas as pd
import torch
from train_model import CustomModel
from transformers import (
    Trainer,
    AlbertTokenizer,
    DataCollatorWithPadding,
    AlbertConfig,
    AutoModel,
)
from datasets import load_metric
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import numpy as np
from preprocess import (
    PreprocessDatasetWithChunks,
    get_word2vec_model,
)  # This is a placeholder, replace with your dataset loading mechanism

MAX_LENGTH = 512
VECTOR_SIZE = 512

# Load the model and tokenizer
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSTIVE": 1}
config = AlbertConfig.from_pretrained(
    pretrained_model_name_or_path="albert-base-v2",
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)
model_path = "models/finalfinalmodel"
model = AutoModel.from_pretrained(model_path)
tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")


def load_config(config_path="config.json"):
    with open(config_path) as config_file:
        return json.load(config_file)


class CustomDataCollatorWithPadding(DataCollatorWithPadding):
    def __init__(
        self,
        tokenizer,
        padding=True,
        max_length=MAX_LENGTH,
        pad_to_multiple_of=None,
        return_tensors="pt",
    ):
        super().__init__(
            tokenizer, padding, max_length, pad_to_multiple_of, return_tensors
        )

    def __call__(self, features):
        batch = super().__call__(features)
        word2vec = [feature["word2vec"] for feature in features]
        word2vec_embeddings = [embedding for embedding in word2vec]

        # tfidf = [feature["tfidf"] for feature in features]
        # tfidf_embeddings = [embedding for embedding in tfidf]

        # Stack all padded embeddings into a single tensor
        batch["word2vec"] = torch.stack(word2vec_embeddings)
        # batch["tfidf"] = torch.stack(tfidf_embeddings)

        return batch


def load_data(filepath):
    return pd.read_csv(filepath).dropna(subset="CONTENT")


config = load_config()
dataframe = load_data(config["annotated_dataset_path"])

tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
word2vec_model = get_word2vec_model(VECTOR_SIZE)
eval_dataset = PreprocessDatasetWithChunks(
    dataframe, tokenizer, word2vec_model, MAX_LENGTH
)

# Initialize the Trainer
trainer = Trainer(model=model)

# Evaluate the model
eval_result = trainer.evaluate(eval_dataset=eval_dataset)
print(eval_result)

# Make predictions
predictions = trainer.predict(test_dataset=eval_dataset)
preds = predictions.predictions.argmax(-1)
true_labels = predictions.label_ids

# Plotting accuracy and loss
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(eval_result["eval_loss"], label="Loss")
ax[0].set_title("Evaluation Loss")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].legend()

# Assuming eval_result contains accuracy, replace "eval_accuracy" with the correct key if necessary
ax[1].plot(eval_result["eval_accuracy"], label="Accuracy")
ax[1].set_title("Evaluation Accuracy")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Accuracy")
ax[1].legend()

plt.tight_layout()
plt.show()

# Confusion Matrix
cm = confusion_matrix(true_labels, preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# ROC Curve (for binary classification)
if np.unique(true_labels).size == 2:  # Check if the task is binary classification
    fpr, tpr, thresholds = roc_curve(true_labels, predictions.predictions[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()
