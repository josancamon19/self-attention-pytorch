# TODO:
# - [ ] train classifier
# - [ ] estimate confidence thresholds
# - [ ] view a full cleaning examples, compare sizes, viz on streamlit, yea

# -- prioritize finishing it
# -- and seeing the data
# -- for some reason this assignment/leaderboard
# -- gives 0 excitement


import streamlit as st
import gzip
import pandas as pd
from fastwarc.warc import ArchiveIterator, WarcRecordType
from cs336_data.a_html import extract_text_from_html_bytes
from cs336_data.b_language_id import language_identification
from cs336_data.c_piid import remove_emails, remove_ip_addresses, remove_phone_numbers
from cs336_data.d_harmful import is_harmful
from cs336_data.e_gopher_heuristics import gopher_filters


@st.cache_data
def load_warc_data(file_path=".data/sample.warc.gz", max_records=100):
    """Load WARC data and apply all filters to see their effects"""
    records_data = []

    count = 0
    with gzip.open(file_path, "rb") as stream:
        for _, record in enumerate(ArchiveIterator(stream)):
            if count >= max_records:
                break

            if record.record_type == WarcRecordType.response:
                url = record.headers.get("WARC-Target-URI", "Unknown URL")
                html_content = record.reader.read()
                plain_text = extract_text_from_html_bytes(html_content)

                # Apply all filters and collect stats
                record_stats = {
                    "id": count + 1,
                    "url": url,
                    "original_text": plain_text[:5000] + "..." if len(plain_text) > 5000 else plain_text,
                    "text_length": len(plain_text),
                }

                # Gopher filter
                gopher_result = gopher_filters(plain_text)
                record_stats["gopher_pass"] = gopher_result["pass_filter"]
                record_stats["gopher_failed_filters"] = [k for k, v in gopher_result["filters"].items() if not v]

                # Language ID
                lang_pass = language_identification(plain_text, True, 0.9)
                record_stats["language_pass"] = lang_pass

                # Harmful content
                harmful_pass = not is_harmful(plain_text, 0.8)
                record_stats["harmful_pass"] = harmful_pass

                # PIID removal
                text_after_emails, email_count = remove_emails(plain_text)
                text_after_ips, ip_count = remove_ip_addresses(text_after_emails)
                text_after_phones, phone_count = remove_phone_numbers(text_after_ips)

                record_stats["email_count"] = email_count
                record_stats["ip_count"] = ip_count
                record_stats["phone_count"] = phone_count
                record_stats["piid_removed_chars"] = len(plain_text) - len(text_after_phones)

                # Overall pass
                overall_pass = gopher_result["pass_filter"] and lang_pass and harmful_pass
                record_stats["overall_pass"] = overall_pass

                records_data.append(record_stats)
                count += 1

    return pd.DataFrame(records_data)


def main():
    st.set_page_config(page_title="WARC Filter Visualization", layout="wide")
    st.title("WARC Data Filter Visualization")
    st.write("Interactive dashboard to see different filters in action")

    # Load data
    with st.spinner("Loading WARC data..."):
        df = load_warc_data()

    st.write(f"Loaded {len(df)} records")

    # Summary stats
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Gopher Pass", f"{df['gopher_pass'].sum()}/{len(df)}")
    with col2:
        st.metric("Language Pass", f"{df['language_pass'].sum()}/{len(df)}")
    with col3:
        st.metric("Not Harmful", f"{(df['harmful_pass']).sum()}/{len(df)}")
    with col4:
        st.metric("Overall Pass", f"{df['overall_pass'].sum()}/{len(df)}")
    with col5:
        st.metric("Avg Text Length", f"{df['text_length'].mean():.0f}")

    # Filter controls
    st.subheader("Filter Records")
    show_only_failed = st.checkbox("Show only records that failed filters")

    if show_only_failed:
        display_df = df[~df["overall_pass"]]
    else:
        display_df = df

    # Record selector
    if len(display_df) > 0:
        selected_idx = st.selectbox(
            "Select record to examine:",
            range(len(display_df)),
            format_func=lambda x: f"Record {display_df.iloc[x]['id']} - {display_df.iloc[x]['url']}",
        )

        record = display_df.iloc[selected_idx]

        # Display record details
        st.subheader(f"Record {record['id']} Details")

        # Create columns for different aspects
        col1, col2 = st.columns([2, 1])

        with col1:
            st.write("**Original Text:**")
            st.text_area("Content", record["original_text"], height=800, key="content")

        with col2:
            st.write("**Filter Results:**")

            # Gopher filter
            gopher_status = "✅ PASS" if record["gopher_pass"] else "❌ FAIL"
            st.write(f"**Gopher Filter:** {gopher_status}")
            if not record["gopher_pass"]:
                st.write("Failed filters:", record["gopher_failed_filters"])

            # Language filter
            lang_status = "✅ PASS" if record["language_pass"] else "❌ FAIL"
            st.write(f"**Language ID:** {lang_status}")

            # Harmful filter
            harmful_status = "✅ PASS" if record["harmful_pass"] else "❌ HARMFUL"
            st.write(f"**Harmful Content:** {harmful_status}")

            # PIID stats
            st.write("**PIID Removal:**")
            st.write(f"- Emails removed: {record['email_count']}")
            st.write(f"- IPs removed: {record['ip_count']}")
            st.write(f"- Phones removed: {record['phone_count']}")
            st.write(f"- Chars removed: {record['piid_removed_chars']}")

            # Overall
            overall_status = "✅ PASS ALL" if record["overall_pass"] else "❌ FILTERED OUT"
            st.write(f"**Overall:** {overall_status}")

    # Show full table
    st.subheader("All Records Summary")

    # Prepare display columns
    display_cols = [
        "id",
        "gopher_pass",
        "language_pass",
        "harmful_pass",
        "email_count",
        "ip_count",
        "phone_count",
        "overall_pass",
        "text_length",
    ]

    st.dataframe(
        display_df[display_cols],
        column_config={
            "gopher_pass": st.column_config.CheckboxColumn("Gopher"),
            "language_pass": st.column_config.CheckboxColumn("Language"),
            "harmful_pass": st.column_config.CheckboxColumn("Not Harmful"),
            "overall_pass": st.column_config.CheckboxColumn("Overall Pass"),
            "text_length": st.column_config.NumberColumn("Text Length"),
        },
    )


if __name__ == "__main__":
    main()
