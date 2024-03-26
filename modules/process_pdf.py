import logging
from typing import Optional
import requests
import fitz
import fitz.utils
import io
import numpy as np

logging.basicConfig(
    filename="logs/debug.txt",
    level=logging.DEBUG,
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def extract_text_by_headings(paper_url: str) -> Optional[tuple[list, list, dict]]:
    """Downloads a PDF from the given URL, analyzes its structure to find headings, extracts text by headings, and annotates the PDF for visual confirmation of identified sections. Returns a tuple containing the table of contents (ToC), a list of section indices, and a dictionary mapping sections to their text.

    Args:
        paper_url (str): URL of the paper PDF to be processed.

    Returns:
        Optional[tuple[list, list, dict]]: A tuple containing the table of contents (list of ToC entries), a list of tuples each representing a section (identified by starting and ending indices on pages), and a dictionary mapping section identifiers to their extracted text, or None if the PDF cannot be processed.
    """
    try:
        response = requests.get(paper_url)
        response.raise_for_status()
        document = fitz.open(stream=io.BytesIO(response.content))
    except Exception as e:
        logging.error(f"Failed to load or process PDF from {paper_url}: {e}")
        return None

    toc = fitz.utils.get_toc(document, simple=False)

    if not toc:
        return None

    heading_indices = {page_num: [] for page_num in range(document.page_count)}
    sections = []

    try:
        for entry in toc:
            page_num = entry[3]["page"]
            try:
                toc_page = document.load_page(page_num)
                page_block_cache = fitz.utils.get_text_blocks(toc_page)
            except ValueError as e:
                logging.error(
                    f"Page {page_num} not in document for URL {paper_url}: {e}"
                )
                continue  # Skip this page and continue with the next one
            blocks = page_block_cache

            point = entry[3]["to"]
            destination_point = fitz.Point(point.x, (toc_page.bound().y1 - point.y))

            distances = []
            for block in blocks:
                distances.append(
                    destination_point.distance_to(fitz.Point((block[0]), (block[1])))
                )
            min_dist_index = np.argmin(distances)
            heading_indices[page_num].append(min_dist_index)

        sections = extract_sections(document, heading_indices)
        toc_block_text = compile_section_text(document, sections)

        return toc, sections, toc_block_text
    except Exception as e:
        logging.error(f"Error processing sections for URL {paper_url}: {e}")


def compile_section_text(
    document: fitz.Document, sections: list[tuple[int, int, int, int]]
) -> dict[int, str]:
    """Compiles the text for each section identified in the PDF document into a dictionary.

    Args:
        document (fitz.Document): The document object being analyzed.
        sections (list[tuple[int, int, int, int]]): A list of tuples representing the sections, each defined by a tuple of section ID, page number, start block index, and end block index.

    Returns:
        dict[int, str]: A dictionary mapping section identifiers to their concatenated text.
    """
    toc_text = {}
    for section_id, page_num, start_block, end_block in sections:
        if section_id not in toc_text:
            toc_text[section_id] = ""
        toc_text[section_id] += " ".join(
            get_block_text(document, page_num, start_block, end_block).split("\n")
        )
    return toc_text


def extract_sections(
    document: fitz.Document, heading_indices: dict[int, list]
) -> list[tuple[int, int, int, int]]:
    """Identifies text sections based on headings found in the document and extracts their indices.

    Args:
        document (fitz.Document): The document object being analyzed.
        heading_indices (dict[int, list]): A dictionary mapping page numbers to lists of indices of heading blocks on those pages.

    Returns:
        list[tuple[int, int, int, int]]: A list of tuples, each representing a section. Each tuple contains the section ID, page number, start block index, and end block index.
    """
    all_headings = sorted(
        [
            (page_num, idx)
            for page_num, indices in heading_indices.items()
            for idx in indices
        ],
        key=lambda x: (x[0], x[1]),
    )
    sections = []

    for i, (page_num, start_idx) in enumerate(all_headings):
        if i + 1 < len(all_headings):
            next_page, next_idx = all_headings[i + 1]
            if next_page == page_num:
                sections.append((i, page_num, start_idx + 1, next_idx))
            else:
                sections += extend_sections_across_pages(
                    document, i, page_num, start_idx + 1, next_page, next_idx
                )
        else:
            sections += extend_sections_to_end(document, i, page_num, start_idx + 1)

    return sections


def extend_sections_across_pages(
    document: fitz.Document,
    section_id: int,
    start_page: int,
    start_idx: int,
    end_page: int,
    end_idx: int,
) -> list[tuple[int, int, int, int]]:
    """Extends the identification of sections across multiple pages when a section spans more than one page.

    Args:
        document (fitz.Document): The document object being analyzed.
        section_id (int): The identifier of the current section.
        start_page (int): The page number where the current section starts.
        start_idx (int): The index of the block where the current section starts.
        end_page (int): The page number where the current section ends.
        end_idx (int): The index of the block where the current section ends.

    Returns:
        list[tuple[int, int, int, int]]: A list of tuples representing the extended sections across pages.
    """
    sections = [
        (
            section_id,
            start_page,
            start_idx,
            len(fitz.utils.get_text(document.load_page(start_page), "blocks")),
        )
    ]
    for page_num in range(start_page + 1, end_page):
        sections.append(
            (
                section_id,
                page_num,
                0,
                len(fitz.utils.get_text(document.load_page(page_num), "blocks")),
            )
        )
    sections.append((section_id, end_page, 0, end_idx))
    return sections


def extend_sections_to_end(
    document: fitz.Document, section_id: int, page_num: int, start_idx: int
) -> list[tuple[int, int, int, int]]:
    """Extends the identification of sections to the end of the document for the last section found.

    Args:
        document (fitz.Document): The document object being analyzed.
        section_id (int): The identifier of the current section.
        page_num (int): The page number where the current section starts.
        start_idx (int): The index of the block where the current section starts.

    Returns:
        list[tuple[int, int, int, int]]: list of tuples representing the extended sections to the end of the document. Each tuple contains the section ID, the page number where the section starts, the index of the starting block on that page, and the total number of blocks on the page, effectively marking the end of the section on that page.
    """
    sections = [
        (
            section_id,
            page_num,
            start_idx,
            len(fitz.utils.get_text(document.load_page(page_num), "blocks")),
        )
    ]
    for next_page in range(page_num + 1, document.page_count):
        sections.append(
            (
                section_id,
                next_page,
                0,
                len(fitz.utils.get_text(document.load_page(page_num), "blocks")),
            )
        )
    return sections


def get_block_text(
    document: fitz.Document, page_num: int, start_block: int, end_block: int
) -> str:
    """Retrieves the concatenated text from specified blocks within a page of the document.

    Args:
        document (fitz.Document): The document object from which text is being extracted.
        page_num (int): The number of the page from which text is being extracted.
        start_block (int): The index of the first block within the page from which to start text extraction.
        end_block (int): The index of the block at which to end text extraction, exclusive.

    Returns:
        str: _description_
    """
    page = document.load_page(page_num)
    blocks = fitz.utils.get_text(page, "blocks")
    text = " ".join(block[4] for block in blocks[start_block:end_block])
    return text
