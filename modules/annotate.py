import json
import logging
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer, util
import tqdm
import os


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
    return pd.read_csv(filepath)


def is_relevant_context(sentence, keywords):
    words = word_tokenize(sentence)
    tagged_words = nltk.pos_tag(words)
    return any(
        word.lower() in keywords and tag.startswith("N") for word, tag in tagged_words
    )


def expand_keywords_with_synonyms(keywords):
    expanded_keywords = set(keywords)
    for keyword in keywords:
        for synset in wn.synsets(keyword):
            for lemma in synset.lemmas():
                expanded_keywords.add(lemma.name().replace("_", " "))
    return expanded_keywords


def aggregate_scores(sentences, model, template_embeddings, expanded_keywords):
    relevant_sentences = [
        sentence
        for sentence in sentences
        if is_relevant_context(sentence, expanded_keywords)
    ]
    if not relevant_sentences:
        return 0
    sentence_embeddings = model.encode(relevant_sentences, convert_to_tensor=True)
    max_similarity = max(
        util.pytorch_cos_sim(sentence_embeddings, template_embeddings).max(1).values
    ).item()
    return max_similarity


def sentence_scores(sentences, model, template_embeddings, expanded_keywords):

    relevant_sentences = [
        sentence
        for sentence in sentences
        if is_relevant_context(sentence, expanded_keywords)
    ]
    if not relevant_sentences:
        return 0
    sentence_embeddings = model.encode(relevant_sentences, convert_to_tensor=True)
    similarity_scores = util.pytorch_cos_sim(sentence_embeddings, template_embeddings)
    scores = [
        (sentence, similarity_scores[i].max().item())
        for i, sentence in enumerate(relevant_sentences)
    ]
    return scores


def classify_and_log_content(content, score, relevance_threshold):
    classification = 1 if score >= relevance_threshold else 0
    if classification == 1:
        logging.debug(f"\nScore: {score}\nContent: {content[:1500]}\n")
    return classification


def main():
    config = load_config()
    initialize_logging(config["log_file_path"])
    df = load_data(config["dataset_path"])
    df = df.dropna(subset=["CONTENT"])
    expanded_keywords = expand_keywords_with_synonyms(config["keywords"])
    model = SentenceTransformer(config["model_name"])
    template_sentences = config["template_sentences"]
    # Ensure this is corrected based on actual config structure
    template_embeddings = model.encode(template_sentences, convert_to_tensor=True)

    annotated_data = []
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), ncols=175, leave=True):
        content = row["CONTENT"]
        if isinstance(content, str):
            sentences = sent_tokenize(content)
            score = aggregate_scores(
                sentences, model, template_embeddings, expanded_keywords
            )
            classification = classify_and_log_content(
                content, score, config["relevance_threshold"]
            )
            annotated_data.append({"CONTENT": content, "LABEL": classification})
        else:
            annotated_data.append({"CONTENT": content, "LABEL": 0})

    # Create a new DataFrame from the annotated data
    annotated_df = pd.DataFrame(annotated_data)
    # Save the annotated dataset to a new CSV file
    annotated_df.to_csv("logs/annotated_dataset.csv", index=False)
    logging.info("Dataset annotation completed and saved.")
    print(
        f"Annotated {annotated_df[annotated_df['LABEL'] == 1].shape[0]} / {annotated_df.shape[0]}"
    )


if __name__ == "__main__":
    nltk.download("averaged_perceptron_tagger")
    nltk.download("punkt")
    nltk.download("wordnet")
    main()
