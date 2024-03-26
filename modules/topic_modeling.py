import pandas as pd
from gensim import corpora, models
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, sent_tokenize
import json

nltk.download("stopwords")


class TopicModeler:
    def __init__(self, config_path="config.json", num_topics=5):
        self.config = self.load_config(config_path)
        self.num_topics = num_topics
        self.tokenizer = RegexpTokenizer(r"\w+")
        self.stop_words = set(stopwords.words("english"))
        self.dictionary = None
        self.lda_model = None

    def load_config(self, config_path):
        with open(config_path) as config_file:
            return json.load(config_file)

    def preprocess_text(self, text):
        """Tokenize, lowercase, and remove stopwords from a single text (paragraph)."""
        tokens = [
            word
            for sent in sent_tokenize(text)
            for word in self.tokenizer.tokenize(sent.lower())
            if word not in self.stop_words
        ]
        return tokens

    def fit_model(self, paragraph):
        """Fit LDA model to the preprocessed paragraph."""
        preprocessed_text = self.preprocess_text(paragraph)
        self.dictionary = corpora.Dictionary([preprocessed_text])
        corpus = [self.dictionary.doc2bow(preprocessed_text)]

        self.lda_model = models.LdaModel(
            corpus=corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            random_state=100,
            update_every=1,
            passes=10,
            alpha="auto",
            eta="auto",
            per_word_topics=True,
        )

    def get_topics_from_context(self, context_window):
        """Extract topics from a specific context window."""
        if not self.lda_model:
            return []

        # Preprocess each sentence in the context window and combine into a single list of tokens
        preprocessed_context = [
            token
            for sentence in context_window
            for token in self.preprocess_text(sentence)
        ]

        # Transform the preprocessed context into the BOW format expected by the model
        bow_context = self.dictionary.doc2bow(preprocessed_context)  # type: ignore

        # Get the topic distribution for the context window
        context_topics = self.lda_model.get_document_topics(
            bow_context, minimum_probability=0.1
        )

        sorted_topics = sorted(context_topics, key=lambda x: x[1], reverse=True)

        # Extract and return words from each sorted topic
        for topic_num, _ in sorted_topics:
            # Extracting words from each topic
            words = self.lda_model.show_topic(topic_num, topn=5)
            # Append the list of words to the topics_words list
            topics_words = [word for word, _ in words]

        return topics_words


# Example usage
if __name__ == "__main__":
    paragraph = """
    Many studies in artificial intelligence overlook the importance of ethical considerations, leading to potential biases.
    Research on climate change often fails to account for local variances in weather patterns, making predictions less accurate.
    While quantum computing holds promise for network security, current models do not fully address quantum-resistant encryption methods.
    """
    context_window = [
        "Research on climate change often fails to account for local variances in weather patterns, making predictions less accurate.",
        "While quantum computing holds promise for network security, current models do not fully address quantum-resistant encryption methods.",
    ]

    topic_modeler = TopicModeler(num_topics=1)
    topic_modeler.fit_model(paragraph)
    topics = topic_modeler.get_topics_from_context(context_window)
    print(topics)
    for i in topics:
        print(i)
