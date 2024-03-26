import logging
import os
import token
import torch
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import punkt, sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from transformers import AlbertTokenizer
import tqdm
from torch.utils.data import Dataset
from gensim.models import Word2Vec, KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import shuffle

nltk.download("punkt", "C:\\Users\\sreer\\Documents\\Learn Pytorch\\.venv\\nltk_data")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
nltk.data.path.append(
    "C:\\Users\\sreer\\Documents\\Learn Pytorch\\.venv\\nltk_data\\punkt"
)

TFIDF_SIZE = 1024
MAX_LENGTH = 512
VECTOR_SIZE = 512

# Initialize logging
logging.basicConfig(
    filename="logs/debug.txt",
    level=logging.DEBUG,
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    errors="backslashreplace",
)

# Load dataset
dataframe = pd.read_csv("logs/annotated_dataset.csv").dropna(subset=["CONTENT"])


def get_word2vec_model(vector_size):
    # Train and save Word2Vec model
    if os.path.exists("models/word2vec_model.model"):
        print("Word2Vec model found. Loading...")
        word2vec_model = KeyedVectors.load("models/word2vec_model.model")
    else:
        print("Word2Vec model not found. Creating...")
        non_null_content = dataframe["CONTENT"].dropna().astype(str).head(10000)
        # Tokenize each paragraph into sentences and flatten the list of lists into a single list of sentences.
        all_sentences = [
            word
            for para in non_null_content
            for sentence in sent_tokenize(para)
            for word in word_tokenize(sentence, language="english")
        ]
        word2vec_model = Word2Vec(
            all_sentences, vector_size=vector_size, window=4, min_count=1, workers=16
        )
        word2vec_model.save("models/word2vec_model.model")
    return word2vec_model


def text_to_word2vec_vectors(text, word2vec_model):
    vectors = []
    for word in word_tokenize(
        text
    ):  # Ensure consistent tokenization with Word2Vec training
        if word in word2vec_model.wv:
            vectors.append(word2vec_model.wv[word])
        else:
            vectors.append(np.zeros(word2vec_model.vector_size))
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)


class PreprocessDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        word2vec_model,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.word2vec_model = word2vec_model
        self.max_length = max_length
        self.data = self._prepare_data(dataframe)

    def _prepare_data(self, dataframe):
        data = []
        for _, row in tqdm.tqdm(
            dataframe.iterrows(), total=dataframe.shape[0], desc="Preparing Data"
        ):
            text = str(row["CONTENT"])
            label = row["LABEL"]

            # Tokenization directly in the class
            tokenized_input = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                padding="max_length",
                truncation=True,
                # max_length=self.max_length,
                return_tensors="pt",
                return_attention_mask=True,
            )

            # Convert the text to a word2vec vector
            word2vec_vector = text_to_word2vec_vectors(text, self.word2vec_model)
            word2vec_tensor = torch.tensor(word2vec_vector, dtype=torch.float)

            data.append(
                {
                    "input_ids": tokenized_input["input_ids"].squeeze(
                        0
                    ),  # Removing the batch dimension
                    "attention_mask": tokenized_input["attention_mask"].squeeze(0),
                    "labels": torch.tensor(label, dtype=torch.long),
                    "word2vec": word2vec_tensor,
                }
            )
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PreprocessDatasetWithChunks(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        word2vec_model,
        # tfidf_vectorizer: TfidfVectorizer,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.word2vec_model = word2vec_model
        # self.tfidf_vectorizer = tfidf_vectorizer
        self.max_length = max_length
        self.data = self._prepare_data(dataframe)
        self.sampled_data = self._balance_dataset()

    # Define the preprocessing function

    # Helper function to convert NLTK POS tags to WordNet POS tags
    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith("J"):
            return wordnet.ADJ
        elif treebank_tag.startswith("V"):
            return wordnet.VERB
        elif treebank_tag.startswith("N"):
            return wordnet.NOUN
        elif treebank_tag.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def preprocess_text(self, text):
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()

        # Tokenize
        tokens = word_tokenize(text.lower())

        # Remove stopwords
        tokens = [
            token for token in tokens if token not in stop_words and token.isalpha()
        ]

        # Part-of-Speech tagging for lemmatization
        pos_tags = nltk.pos_tag(tokens)

        # Lemmatize and stem tokens
        lemmatized_tokens = [
            lemmatizer.lemmatize(token, pos=self.get_wordnet_pos(pos_tag))
            for token, pos_tag in pos_tags
        ]
        stemmed_tokens = [stemmer.stem(token) for token in lemmatized_tokens]

        return " ".join(stemmed_tokens)

    def _prepare_data(self, dataframe):
        data = []
        for _, row in tqdm.tqdm(
            dataframe.iterrows(), total=dataframe.shape[0], desc="Preparing Data"
        ):
            text = str(row["CONTENT"])
            processed_text = self.preprocess_text(text)
            label = row["LABEL"]

            # Basic tokenization without splitting into max_length
            basic_tokenized_input = self.tokenizer.encode_plus(
                processed_text,
                add_special_tokens=True,
                return_tensors="pt",
                return_attention_mask=True,
                truncation=False,
            )

            # Convert the text to a word2vec vector
            word2vec_vector = text_to_word2vec_vectors(text, self.word2vec_model)
            word2vec_tensor = torch.tensor(word2vec_vector, dtype=torch.float)

            # Vectorize the text into TFIDF vectors
            # tfidf_vector = self.tfidf_vectorizer.transform([text]).toarray()[0]
            # tfidf_tensor = torch.tensor(tfidf_vector, dtype=torch.float)

            # Check if tokenized input needs to be chunked
            total_length = basic_tokenized_input["input_ids"].size(1)
            if total_length > self.max_length:
                # Calculate the number of chunks needed
                num_chunks = total_length // self.max_length + int(
                    total_length % self.max_length != 0
                )

                for chunk_idx in range(num_chunks):
                    start_idx = chunk_idx * self.max_length
                    end_idx = start_idx + self.max_length

                    # Chunking the input_ids and attention_mask
                    chunk_input_ids = basic_tokenized_input["input_ids"][
                        :, start_idx:end_idx
                    ]
                    chunk_attention_mask = basic_tokenized_input["attention_mask"][
                        :, start_idx:end_idx
                    ]

                    data.append(
                        {
                            "input_ids": chunk_input_ids.squeeze(
                                0
                            ),  # Remove batch dimension
                            "attention_mask": chunk_attention_mask.squeeze(0),
                            "labels": torch.tensor(label, dtype=torch.long),
                            "word2vec": word2vec_tensor,
                            # "tfidf": tfidf_tensor,
                        }
                    )
            else:
                # If no chunking is needed, use the basic_tokenized_input as it is
                data.append(
                    {
                        "input_ids": basic_tokenized_input["input_ids"].squeeze(0),
                        "attention_mask": basic_tokenized_input[
                            "attention_mask"
                        ].squeeze(0),
                        "labels": torch.tensor(label, dtype=torch.long),
                        "word2vec": word2vec_tensor,
                        # "tfidf": tfidf_tensor,
                    }
                )
        return data

    def _balance_dataset(self):
        sampled_data = []

        # Convert labels into a numpy array for easy processing
        word2vec = np.array([item["word2vec"].numpy() for item in self.data])
        labels = [item["labels"].item() for item in self.data]

        # tfidf = np.array(item["tfidf"].item() for item in self.data)

        # Counts of each class to determine oversampling and undersampling targets
        unique, counts = np.unique(labels, return_counts=True)
        class_counts = dict(zip(unique, counts))

        # Target counts after balancing
        target_minority_count = 2 * class_counts[1]  # Double the minority class
        target_majority_count = class_counts[
            1
        ]  # Match the majority class to original minority size

        # Define sampling strategies
        oversample_strategy = {1: target_minority_count}
        undersample_strategy = {0: target_majority_count}

        ros = RandomOverSampler(sampling_strategy={1: 3000}, random_state=42)
        rus = RandomUnderSampler(sampling_strategy={0: 3000}, random_state=42)
        input_o, labels_o = ros.fit_resample(word2vec, labels)
        input_u, labels_u = rus.fit_resample(word2vec, labels)
        o_indices = ros.sample_indices_.tolist()
        u_indices = rus.sample_indices_.tolist()

        sampled_data = [self.data[idx] for idx in o_indices] + [
            self.data[idx] for idx in u_indices
        ]
        print(f"Total dataset size -> {len(sampled_data)}")
        pd.DataFrame(
            sampled_data, columns=["input_ids", "attention_mask", "labels", "word2vec"]
        ).to_csv("logs/preprocessed_database.csv")
        return sampled_data

    def __len__(self):
        return len(self.sampled_data)

    def __getitem__(self, idx):
        return self.sampled_data[idx]


if __name__ == "__main__":
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
    print(
        PreprocessDatasetWithChunks(
            dataframe,
            tokenizer,
            get_word2vec_model(vector_size=VECTOR_SIZE),
        )
    )
    pass
