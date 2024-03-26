from flask import Flask, request, render_template
import pandas as pd
from transformers import pipeline
from modules import fetch_arxiv, process_pdf
import json

app = Flask(__name__)

# Ensure you have the arXiv dataset in the same directory or adjust the path accordingly
dataset_path = "arxiv-metadata-oai-snapshot.json"
fetcher = fetch_arxiv.ArxivMetadataFetcher(dataset_path, max_results=20)

# Placeholder for YourPretrainedModel import
# from your_model_module import YourPretrainedModel


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        search_query = request.form.get("search_query", "")
        papers = fetcher.fetch_papers(start_year=2022, search_queries=[search_query])
        summaries = process_and_summarize(papers)
        return render_template("summary_results.html", summaries=summaries)
    return render_template("index.html")


def load_config(config_path="config.json"):
    with open(config_path) as config_file:
        return json.load(config_file)


def process_paper(paper):
    link = paper["link"]
    extraction_result = process_pdf.extract_text_by_headings(link)
    if extraction_result:
        _, _, toc_block_text = extraction_result
        dataset_entries = [(key, content) for key, content in toc_block_text.items()]
        return dataset_entries
    return None


def create_dataframe(papers):
    data = []
    for paper in papers:
        processed_data = process_paper(paper)
        if processed_data:
            for heading, content in processed_data:
                data.append({"Heading": heading, "Content": content})
    return pd.DataFrame(data)


def label_dataframe(df):
    # Placeholder for actual model prediction
    # Assuming all content is relevant for demonstration purposes
    df["Label"] = 1
    return df


if __name__ == "__main__":
    app.run(debug=True, port=5100)
