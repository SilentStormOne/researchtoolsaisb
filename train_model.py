import logging
import pandas as pd
from transformers import (
    AlbertModel,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    pipeline,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.process_pdf import extract_text_by_headings
from preprocess import (
    get_word2vec_model,
    PreprocessDatasetWithChunks,
)
import numpy as np
import evaluate
from sklearn.model_selection import train_test_split
import optuna
import os
import json

MAX_LENGTH = 512
VECTOR_SIZE = 512
TFIDF_SIZE = 1024


def load_config(config_path="config.json"):
    with open(config_path) as config_file:
        return json.load(config_file)


def initialize_logging(log_file_path):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    logging.basicConfig(
        filename=log_file_path,
        level=logging.DEBUG,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        errors="backslashreplace",
    )
    # Test logging at various levels
    logging.debug("Debugging log initialized.")
    logging.info("Info level logging initialized.")


def load_data(filepath):
    return pd.read_csv(filepath).dropna(subset="CONTENT")


class CustomModel(AlbertForSequenceClassification):
    def __init__(self, config, word2vec_size):
        super().__init__(config)
        self.word2vec_size = word2vec_size
        # self.tfidf_size = tfidf_size
        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Define attention layers
        self.attention_query = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_key = nn.Linear(word2vec_size, config.hidden_size)
        self.attention_value = nn.Linear(word2vec_size, config.hidden_size)
        self.attention_scale = 1 / (config.hidden_size**0.5)

        # Additional dense layers
        self.dense1 = nn.Linear(
            config.hidden_size * 2, 512
        )  # Example dimension, adjust as needed
        self.dense2 = nn.Linear(512, 128)  # Example dimension, adjust as needed

        # Final classifier
        self.classifier = nn.Linear(128, config.num_labels)

        # Initialize weights of the newly added layers
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        word2vec=None,
        # tfidf=None,
    ):
        # Output from the original ALBERT model
        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[1]

        # Prepare for attention mechanism
        query = self.attention_query(pooled_output)
        key = self.attention_key(word2vec)  # type: ignore
        value = self.attention_value(word2vec)  # type: ignore

        # Compute attention scores and apply softmax
        attention_scores = (
            torch.matmul(query, key.transpose(-2, -1)) * self.attention_scale
        )
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value)

        # Concatenate ALBERT pooled output with attention output
        combined_embeddings = torch.cat((pooled_output, attention_output), dim=1)
        combined_embeddings = self.dropout(combined_embeddings)

        # Pass through additional dense layers with activation functions
        x = F.relu(self.dense1(combined_embeddings))
        x = F.relu(self.dense2(x))

        # Pass through the classifier
        logits = self.classifier(x)

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    #  For regression
                    loss_fct = nn.MSELoss()
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    # For classification
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                raise ValueError("You must define the problem_type")

        # Return the usual output expected by Hugging Face's Trainer
        return ((loss,) + (logits,)) if loss is not None else (logits,)


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


def optuna_hp_space(trial: optuna.Trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True),
    }


def main():
    """
    ASSIGNING DATA AND LOADING MODELS
    """
    config = load_config()
    dataframe = load_data(config["annotated_dataset_path"])

    # Initialize the TfidfVectorizer
    # tfidf_vectorizer = TfidfVectorizer(
    #     stop_words="english",  # Remove common 'stop words'
    #     max_features=TFIDF_SIZE,  # Limit the number of features to the top 10,000
    #     ngram_range=(1, 2),  # Consider both single words and bigrams
    #     max_df=0.6,  # Ignore terms that appear in more than 30% of the documents
    #     min_df=10,  # Ignore terms that appear in less than 2 documents
    # )
    # print("Fitting vectorizer on text data.")
    # Fit the vectorizer on the text data
    filtered_df = dataframe[dataframe["LABEL"] == 1]
    # tfidf_vectorizer.fit(filtered_df["CONTENT"])

    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSTIVE": 1}
    config = AlbertConfig.from_pretrained(
        pretrained_model_name_or_path="albert-base-v2",
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )
    model = CustomModel(config=config, word2vec_size=VECTOR_SIZE)

    # Function for conducting optuna hyperparam search.
    def model_init(trial):
        return CustomModel(config=config, word2vec_size=MAX_LENGTH)

    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
    word2vec_model = get_word2vec_model(VECTOR_SIZE)
    preprocessed_dataset = PreprocessDatasetWithChunks(
        dataframe, tokenizer, word2vec_model, MAX_LENGTH
    )
    y = np.array(
        [preprocessed_dataset[i]["labels"] for i in range(len(preprocessed_dataset))]
    )

    """
    EVALUATION FUNCTIONS AND VARIABLES
    """
    training_dataset, eval_dataset = train_test_split(
        preprocessed_dataset, test_size=0.2
    )
    data_collator = CustomDataCollatorWithPadding(tokenizer)

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    """
    TRAINING AND EVALUATING
    """

    training_args = TrainingArguments(
        output_dir="models/mymodel",
        overwrite_output_dir=True,
        dataloader_num_workers=8,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=6,
        weight_decay=0.01,
        save_strategy="steps",
        save_steps=0.5,
        evaluation_strategy="epoch",
        save_safetensors=True,
        save_only_model=True,
        fp16=True,
        warmup_steps=200,
    )

    trainer = Trainer(
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        model_init=model_init,  # type: ignore
        compute_metrics=compute_metrics,  # type: ignore
    )

    trainer.train()

    best_trial = trainer.hyperparameter_search(
        direction="minimize",
        hp_space=optuna_hp_space,
        backend="optuna",
        n_trials=10,
    )

    return best_trial


if __name__ == "__main__":
    config = load_config()
    initialize_logging(config["log_file_path"])

    best_trial = main()
    print(
        f"\nBest trial from optuna study: {best_trial}\n{best_trial.objective}\n{best_trial.run_id}\n{best_trial.run_summary}\n{best_trial.hyperparameters}\n\n"
    )

    # classifier = pipeline("text-classification", model=model)
    # extraction_result = extract_text_by_headings("https://arxiv.org/pdf/2403.00189.pdf")
    # if extraction_result:
    #     toc = extraction_result[0]
    #     toc_block_text = extraction_result[2]
    #     for key in toc_block_text:
    #         pred_output = classifier(toc_block_text[key], num_workers=4)
    #         print(f"Predicted label for key {toc[key][1]}: {pred_output}")
    # else:
    #     print("\nThe extract_text_by_headings() function has failed.")
