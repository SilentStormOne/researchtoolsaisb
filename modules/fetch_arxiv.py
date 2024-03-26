import json
import regex
import datetime
from tqdm import tqdm


class ArxivMetadataFetcher:
    def __init__(self, dataset_path, max_results=100):
        self.dataset_path = dataset_path
        self.max_results = max_results

    def get_metadata(self):
        with open(self.dataset_path, "r") as f:
            for line in f:
                yield line

    def check_search_query(self, text: str, queries):
        for query in queries:
            pattern = regex.compile(rf"{query}")
            if pattern.search(string=text) is not None:
                return True
        return False

    def fetch_papers(self, start_year: int, search_queries: list):
        papers = []
        metadata = self.get_metadata()

        print("Fetching papers from arXiv...")
        progress_bar = tqdm(range(self.max_results), unit="papers", ncols=100)

        for paper in metadata:
            if len(papers) >= self.max_results:
                break

            paper_dict = json.loads(paper)
            try:
                title = paper_dict.get("title")
                abstract = paper_dict.get("abstract")
                paper_id = paper_dict.get("id")
                link = f"https://arxiv.org/pdf/{paper_id}"

                # Variable to check.
                year = datetime.datetime.strptime(
                    paper_dict.get("versions")[0]["created"], "%a, %d %b %Y %H:%M:%S %Z"
                ).strftime("%Y")

                if int(year) >= start_year and (
                    self.check_search_query(title, search_queries)
                    or self.check_search_query(abstract, search_queries)
                ):
                    progress_bar.update(1)
                    papers.append({"title": title, "abstract": abstract, "link": link})
            except Exception as e:
                print(f"Could not fetch metadata: {e}")
                pass

        return papers


# Example usage
if __name__ == "__main__":
    dataset_path = "arxiv-metadata-oai-snapshot.json"
    fetcher = ArxivMetadataFetcher(dataset_path, max_results=1000)
    search_queries = ["literature review"]
    papers = fetcher.fetch_papers(start_year=2022, search_queries=search_queries)
    for paper in papers:
        print(paper)
    print(json.dumps(papers, indent=4))
