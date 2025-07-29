import gzip
from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding


def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    try:
        html_string = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        detected_encoding = detect_encoding(html_bytes)
        if detected_encoding:
            try:
                html_string = html_bytes.decode(detected_encoding)
            except (UnicodeDecodeError, LookupError):
                html_string = html_bytes.decode("utf-8", errors="replace")
        else:
            html_string = html_bytes.decode("utf-8", errors="replace")
    return extract_plain_text(html_string)
