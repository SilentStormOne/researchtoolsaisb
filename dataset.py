import logging
import pandas
import regex
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from modules.fetch_arxiv import ArxivMetadataFetcher
from modules.process_pdf import extract_text_by_headings
from modules.fetch_ieee import fetch_ieee_data
from tqdm import tqdm
import nltk

# Set up logging
logging.basicConfig(
    filename="logs/debug.txt",
    level=logging.DEBUG,
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_nltk_resources():
    nltk.download("punkt")
    nltk.download("wordnet")
    nltk.download("stopwords")
    # Force load the actual data
    nltk.corpus.wordnet.ensure_loaded()
    nltk.data.find("tokenizers/punkt")
    # Load stopwords for the first time to ensure it's available in subsequent uses
    nltk.corpus.stopwords.words("english")


def process_paper(paper, callback=None):
    title = paper["title"]
    abstract = paper["abstract"]
    link = paper["link"]

    logging.debug(f"\nTitle: {title}\nAbstract: {abstract}\nLink: {link}\n\n")
    # Assuming the existence of a function to extract text by headings from the PDF link
    extraction_result = extract_text_by_headings(link)
    if extraction_result:
        _, _, toc_block_text = extraction_result

        # Auto-labelling dataset for testing purposes.
        dataset_entries = []
        for key, content in toc_block_text.items():
            dataset_entries.append((key, content))
        if callback:
            callback()
        return dataset_entries
    else:
        logging.debug(
            f"Table of contents not found or error in processing for: {title}\n"
        )
        if callback:
            callback()
        return []


def dataset():
    # Initialize the ArxivMetadataFetcher with the dataset path and maximum number of results
    dataset_path = "arxiv-metadata-oai-snapshot.json"
    # Define the search queries, can be multiple as works using OR logic. Then fetching papers from query.
    fetcher = ArxivMetadataFetcher(dataset_path, max_results=2000)
    search_queries = ["literature review", "review", "limitation"]
    papers = fetcher.fetch_papers(start_year=2021, search_queries=search_queries)

    print("Creating raw text dataset...")
    progress_bar = tqdm(range(len(papers)), unit=" papers", ncols=150, leave=False)

    def update_progress_bar():
        progress_bar.update(1)

    with ThreadPoolExecutor(max_workers=32) as executor:
        dataset = []
        futures = []
        for paper in papers:
            futures.append(executor.submit(process_paper, paper, update_progress_bar))
        for future in as_completed(futures):
            dataset.extend(future.result())

    # Convert the dataset to a DataFrame and save to CSV
    progress_bar.close()
    pandas.options.display.max_rows = 10000
    dataset_df = pandas.DataFrame(dataset, columns=["HEADING", "CONTENT"])
    dataset_df.to_csv(path_or_buf="logs/dataset.csv", index=False)
    print(f"\n{dataset_df}")

    return dataset_df


if __name__ == "__main__":
    load_nltk_resources()
    dataset()
