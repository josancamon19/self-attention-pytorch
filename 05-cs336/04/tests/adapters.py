from __future__ import annotations

import os
from typing import Any
from cs336_data.a_html import extract_text_from_html_bytes
from cs336_data.b_language_id import language_identification
from cs336_data.c_piid import remove_emails, remove_ip_addresses, remove_phone_numbers
from cs336_data.d_harmful import check_nsfw, check_hatespeech
from cs336_data.e_gopher_heuristics import gopher_filters
from cs336_data.g_deduplication import exact_deduplication, minhash_deduplication
from cs336_data.f_quality_classifier import classify_quality


def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    return extract_text_from_html_bytes(html_bytes)


def run_identify_language(text: str) -> tuple[Any, float]:
    return language_identification(text)


def run_mask_emails(text: str) -> tuple[str, int]:
    return remove_emails(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    return remove_phone_numbers(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    return remove_ip_addresses(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    return check_nsfw(text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    return check_hatespeech(text)


def run_classify_quality(text: str) -> tuple[Any, float]:
    return classify_quality(text)


def run_gopher_quality_filter(text: str) -> bool:
    return gopher_filters(text)["pass_filter"]


def run_exact_line_deduplication(input_files: list[os.PathLike], output_directory: os.PathLike):
    return exact_deduplication(input_files, output_directory)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    return minhash_deduplication(
        input_files,
        num_hashes,
        num_bands,
        ngrams,
        jaccard_threshold,
        output_directory,
    )
