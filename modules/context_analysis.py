import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer, util


class TextProcessor:
    def __init__(self, config_path="config.json"):
        self.config = self.load_config(config_path)
        self.model = SentenceTransformer(self.config["model_name"])
        self.expanded_keywords = self.expand_keywords_with_synonyms(
            self.config["keywords"]
        )
        self.template_embeddings = self.model.encode(
            self.config["template_sentences"], convert_to_tensor=True
        )
        # Initialize scores and sentences lists
        self.scores = []
        self.sentences = []

    def load_config(self, config_path):
        with open(config_path) as config_file:
            return json.load(config_file)

    def is_relevant_context(self, sentence):
        words = word_tokenize(sentence)
        tagged_words = nltk.pos_tag(words)
        return any(
            word.lower() in self.expanded_keywords and tag.startswith("N")
            for word, tag in tagged_words
        )

    def expand_keywords_with_synonyms(self, keywords):
        expanded_keywords = set(keywords)
        for keyword in keywords:
            for synset in wn.synsets(keyword):
                for lemma in synset.lemmas():  # type: ignore
                    expanded_keywords.add(lemma.name().replace("_", " "))
        return expanded_keywords

    def process_text(self, text):
        self.sentences = sent_tokenize(text)
        self.scores = self.sentence_scores(self.sentences)
        return self.scores

    def sentence_scores(self, sentences):
        relevant_sentences = [
            sentence for sentence in sentences  # if self.is_relevant_context(sentence)
        ]
        if not relevant_sentences:
            return []
        sentence_embeddings = self.model.encode(
            relevant_sentences, convert_to_tensor=True
        )
        similarity_scores = util.pytorch_cos_sim(
            sentence_embeddings, self.template_embeddings  # type: ignore
        )
        scores = [
            (sentence, similarity_scores[i].max().item())
            for i, sentence in enumerate(relevant_sentences)
        ]
        return scores

    def filter_sentences_by_threshold(self, threshold):
        return [sentence for sentence, score in self.scores if score > threshold]

    def max_score_context_window(self, text):
        # Process the text to get scored sentences
        scored_sentences = self.process_text(text)
        if not scored_sentences:
            return []
        # Find the index of the sentence with the maximum score
        max_score_idx = max(
            range(len(scored_sentences)), key=lambda i: scored_sentences[i][1]
        )
        # Calculate start and end indices for the context window
        start_idx = max(0, max_score_idx - 2)
        end_idx = min(len(scored_sentences), max_score_idx + 3)  # exclusive end index
        # Return the context window sentences
        return [scored_sentences[i][0] for i in range(start_idx, end_idx)]


if __name__ == "__main__":
    text_processor = TextProcessor()
    text = "Non-Abelian anyons have garnered extensive attention for obeying exotic non-Abelian statistics and potential applications to fault-tolerant quantum computation. Although the prior research has predominantly focused on non-Abelian statistics without the necessity of symmetry protection, recent progresses have shown that symmetries can play essential roles and bring about a notion of the symmetry-protected non-Abelian (SPNA) statistics. In this work, we extend the concept of SPNA statistics to strongly-correlated systems which host parafermion zero modes (PZMs). This study involves a few fundamental results proved here. First, we unveil a generic unitary symmetry mechanism that protects PZMs from local couplings. Then, with this symmetry protection, the PZMs can be categorized into two nontrivial sectors, each maintaining its own parity conservation, even though the whole system cannot be dismantled into separate subsystems due to nonlinear interactions. Finally, by leveraging the parity conservation of each sector and the general properties of the effective braiding Hamiltonian, we prove rigorously that the PZMs intrinsically obey SPNA statistics. To further confirm the results, we derive the braiding matrix at a tri-junction. In addition, we propose a physical model that accommodates a pair of PZMs protected by mirror symmetry and satisfying the generic theory. This work shows a broad spectrum of strongly-correlated systems capable of hosting fractional SPNA quasiparticles and enriches our comprehension of fundamental quantum statistics linked to the symmetries that govern the exchange dynamics. "
    threshold = 0.1
    processed_text = text_processor.process_text(text)
    print(processed_text)
