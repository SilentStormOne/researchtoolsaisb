import logging
import modules.xploreapi as xapi

logging.basicConfig(
    filename="logs/debug.txt",
    level=logging.DEBUG,
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def fetch_ieee_data(api_key="zvezy48k2yvxsngfbgewx7gs"):
    """
    Fetches data from the IEEE Xplore API using filters.

    Parameters:
    - api_key: API key for authenticating requests to the IEEE Xplore API.

    Returns:
    - The API response data.
    """
    query = xapi.XPLORE(api_key)
    query.resultsFilter("open_access", "True")
    query.resultsFilter("start_year", "2024")
    query.maximumResults(1)
    query.dataType("xml")
    query.abstractText("review")
    query.articleTitle("review of emotion recognition")
    return query.callAPI()
