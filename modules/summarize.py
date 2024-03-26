import pandas as pd
from context_analysis import TextProcessor
from topic_modeling import TopicModeler
from transformers import pipeline


class PaperSummarizer:
    def __init__(self, config="config.json", num_topics=5):
        """
        Initializes the paper summarizer with necessary configurations.

        Parameters:
        - text_processor_config: Path to the configuration file for the TextProcessor.
        - topic_modeler_config: Path to the configuration file for the TopicModeler.
        - num_topics: Number of topics to use in the TopicModeler.
        """
        self.text_processor = TextProcessor(config)
        self.topic_modeler = TopicModeler(config, num_topics=num_topics)
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    def summarize(self, paragraphs):
        """
        Summarizes a single paper based on input paragraphs.

        Parameters:
        - paragraphs: A list of strings, where each string is a paragraph from the paper.

        Returns:
        - A string containing the summary of the input paper.
        """
        context_analyzed_text = []
        all_topics = []

        for paragraph in paragraphs:
            # Use max_score_context_window to extract relevant sentences
            processed_text = self.text_processor.max_score_context_window(paragraph)
            context_analyzed_text.extend(processed_text)

            # Fit the model to the paragraph and extract topics from the context window
            self.topic_modeler.fit_model(paragraph)
            topics = self.topic_modeler.get_topics_from_context(processed_text)
            all_topics.extend(topics)

        # Combine extracted sentences and topics for summarization input
        summarization_input = " ".join(
            ["Topics"] + [f"{topic}, " for topic in all_topics] + context_analyzed_text
        )
        summarization_input = (
            "Prepare a summary to detail limitations, challenges and problems in the following topics and paragraphs: "
            + summarization_input
        )
        # print(summarization_input)

        # Generate the summary
        summary = self.summarizer(
            summarization_input, max_length=1000, min_length=100, do_sample=False
        )
        return summary[0]["summary_text"]


# Example usage
if __name__ == "__main__":
    # Assuming paragraphs is a list of strings, each representing a paragraph of the paper.
    df = pd.read_csv("logs/annotated_dataset.csv")
    paragraphs = df[df["LABEL"] == 1]["CONTENT"][10:17]

    summarizer = PaperSummarizer()
    summary = summarizer.summarize(paragraphs)
    print(summary)
