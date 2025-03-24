import os
import time
import re
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import pdfplumber  # for PDF extraction
from groq import Groq
from io import BytesIO

# Create a folder to save outputs
RESULTS_FOLDER = "results"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

################################
# 1. EXTRACTION CODE
################################

def remove_page_lines(text):
    """Removes any page-number lines like 'Page 3 of 2765'."""
    return re.sub(r'Page\s+\d+\s+of\s+\d+', '', text)

def remove_quoted_text(text):
    """Removes quoted text starting with a line like 'On mm/dd/yyyy at hh:mm AM/PM, ... wrote:'."""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        if re.search(r'^\s*On\s+\d{1,2}/\d{1,2}/\d{4}\s+at\s+.*\swrote:', line):
            break
        cleaned.append(line)
    return "\n".join(cleaned).strip()

def parse_single_message(block_text):
    """
    Parses a single message block into header fields and message text.
    Expected header lines:
      Sent:, From:, To:, Subject:, (Attachments: skipped)
    """
    block_text = remove_page_lines(block_text)
    lines = [line.strip() for line in block_text.splitlines() if line.strip()]
    msg_data = {
        "Sent Time": "",
        "Sender": "",
        "Receiver": "",
        "Subject": "",
        "Message Text": ""
    }
    body_lines = []
    header_finished = False
    for line in lines:
        if not header_finished:
            if line.startswith("Sent:"):
                msg_data["Sent Time"] = line[len("Sent:"):].strip()
            elif line.startswith("From:"):
                msg_data["Sender"] = line[len("From:"):].strip()
            elif line.startswith("To:"):
                receiver_line = re.sub(r'\(.*?\)', '', line[len("To:"):]).strip()
                msg_data["Receiver"] = receiver_line
            elif line.startswith("Subject:"):
                msg_data["Subject"] = line[len("Subject:"):].strip()
            elif line.startswith("Attachments:"):
                continue
            else:
                header_finished = True
                body_lines.append(line)
        else:
            body_lines.append(line)
    body_text = "\n".join(body_lines).strip()
    body_text = remove_quoted_text(body_text)
    msg_data["Message Text"] = body_text
    return msg_data

def process_pdf_file(pdf_path):
    """
    Opens the PDF using pdfplumber, extracts text, splits by 'Message X of Y' markers,
    and returns a list of message dictionaries.
    """
    all_pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                all_pages_text.append(txt)
    full_text = "\n".join(all_pages_text)
    parts = re.split(r'(Message\s+\d+\s+of\s+\d+)', full_text)
    messages = []
    for i in range(1, len(parts)-1, 2):
        msg_header = parts[i].strip()  # e.g., "Message 5 of 1380"
        block_text = parts[i+1].strip()
        if not block_text:
            continue
        m = re.search(r'Message\s+(\d+)\s+of\s+\d+', msg_header)
        message_number = m.group(1) if m else ""
        msg_data = parse_single_message(block_text)
        msg_data["Message Number"] = message_number
        messages.append(msg_data)
    return messages

def save_to_excel(messages, excel_path):
    """
    Saves the messages to Excel with specified columns.
    """
    df = pd.DataFrame(messages, columns=[
        "Message Number",
        "Sent Time",
        "Sender",
        "Receiver",
        "Subject",
        "Message Text"
    ])
    df.to_excel(excel_path, index=False)
    st.success(f"Saved {len(messages)} messages to {excel_path}")

################################
# 2. GROQ API & CLASSIFICATION FUNCTIONS
################################

client = Groq(api_key="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")

def classify_message_groq(message_text):
    """
    Calls the Groq API to analyze message text for manipulative behavior and sentiment.
    Returns classification text.
    """
    if not isinstance(message_text, str):
        message_text = str(message_text)
    prompt = (
        "Analyze the following message for signs of manipulative behavior and assess its sentiment. "
        "Classify it into one or more of the following categories: Coercive Control, Gaslighting, False Accusations, "
        "Emotional Manipulation, Guilt-Tripping, Self-Advocating, Cooperative, Emotional Overwhelm. "
        "Also indicate whether the overall sentiment is positive, neutral, or negative, and provide a brief explanation.\n\n"
        f"Message: {message_text}"
    )
    messages = [
        {"role": "system", "content": "You are an expert in analyzing psychological abuse and sentiment in digital communications."},
        {"role": "user", "content": prompt}
    ]
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.6,
        max_completion_tokens=4096,
        top_p=0.95,
        stream=True,
        stop=None
    )
    result = ""
    for chunk in completion:
        result += (chunk.choices[0].delta.content or "")
    return result

def parse_categories(classification_text):
    """
    Parses the classification text to extract categories.
    """
    categories = [
        "Coercive Control", "Gaslighting", "False Accusations",
        "Emotional Manipulation", "Guilt-Tripping",
        "Self-Advocating", "Cooperative", "Emotional Overwhelm"
    ]
    found = []
    for cat in categories:
        if cat.lower() in classification_text.lower():
            found.append(cat)
    return found

################################
# 3. FINAL REPORT GENERATION
################################

def generate_final_report(summary_text):
    """
    Generates a final aggregated report via Groq API.
    """
    prompt = (
        "Based on the following aggregated analysis of custody communications, "
        "produce a final comprehensive report suitable for legal review. The report should be concise, "
        "objective, and clearly summarize the key patterns of manipulative behavior detected in the communications.\n\n"
        + summary_text
    )
    messages = [
        {"role": "system", "content": "You are an expert in summarizing analytical reports for legal contexts."},
        {"role": "user", "content": prompt}
    ]
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.6,
        max_completion_tokens=4096,
        top_p=0.95,
        stream=True,
        stop=None
    )
    report = ""
    for chunk in completion:
        report += (chunk.choices[0].delta.content or "")
    return report

def generate_report(summary_text):
    """Wrapper for generate_final_report."""
    return generate_final_report(summary_text)

################################
# 4. SEPARATE MESSAGES BY SENDER
################################

def separate_by_sender(messages):
    """
    Splits messages into two sets based on sender name.
    """
    parent_a_msgs = []  # e.g., messages from Barbara
    parent_b_msgs = []  # e.g., messages from Shola
    for msg in messages:
        sender_name = msg["Sender"].lower()
        if "barbara" in sender_name:
            parent_a_msgs.append(msg)
        else:
            parent_b_msgs.append(msg)
    return parent_a_msgs, parent_b_msgs

################################
# 5. END-TO-END PROCESSING FUNCTIONS
################################

def run_extraction(pdf_file):
    """
    Runs extraction on the provided PDF file and returns a DataFrame.
    """
    msgs = process_pdf_file(pdf_file)
    df = pd.DataFrame(msgs)
    return df

def run_classification(df):
    """
    Iterates over messages, classifies each using Groq API,
    and adds classification and parsed categories.
    """
    classifications = []
    for idx, row in df.iterrows():
        st.write(f"Classifying message {idx+1}/{len(df)} from sender: {row['Sender']}...")
        classification = classify_message_groq(row["Message Text"])
        classifications.append(classification)
        time.sleep(1)  # Avoid rate limits
    df["Classification"] = classifications
    df["Parsed Categories"] = df["Classification"].apply(parse_categories)
    return df

def generate_aggregated_summary(df):
    """
    Generates an aggregated summary text from the classified messages.
    """
    total = len(df)
    cat_counts = {}
    for cats in df["Parsed Categories"]:
        for cat in cats:
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
    summary = f"Total messages: {total}\n\nCategory frequencies:\n"
    for cat, cnt in cat_counts.items():
        summary += f"  - {cat}: {cnt}\n"
    return summary

################################
# 6. DETAILED VISUALIZATION FUNCTIONS
################################

def plot_messages_per_day(df, save_path=None):
    """Plots a timeline chart showing messages per day."""
    try:
        df["Timestamp"] = pd.to_datetime(df["Sent Time"], errors='coerce')
    except Exception:
        df["Timestamp"] = pd.date_range(start="2024-01-01", periods=len(df), freq="H")
    df["Date"] = df["Timestamp"].dt.date
    daily_counts = df.groupby("Date").size()
    fig, ax = plt.subplots()
    ax.plot(daily_counts.index, daily_counts.values, marker="o")
    ax.set_xlabel("Date")
    ax.set_ylabel("Message Count")
    ax.set_title("Messages per Day")
    plt.xticks(rotation=45)
    if save_path:
        fig.savefig(save_path)
    st.pyplot(fig)

def plot_category_stacked_by_sender(df, save_path=None):
    """Creates a stacked bar chart of category distribution by sender."""
    df_exploded = df.explode("Parsed Categories")
    cat_counts = df_exploded.groupby(["Sender", "Parsed Categories"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    cat_counts.plot(kind="bar", stacked=True, ax=ax)
    ax.set_xlabel("Sender")
    ax.set_ylabel("Message Count")
    ax.set_title("Stacked Category Distribution by Sender")
    if save_path:
        fig.savefig(save_path)
    st.pyplot(fig)

def plot_specific_categories_over_time(df, category, save_path=None):
    """Plots a line chart showing daily frequency of a specific category."""
    try:
        df["Timestamp"] = pd.to_datetime(df["Sent Time"], errors='coerce')
    except Exception:
        df["Timestamp"] = pd.date_range(start="2024-01-01", periods=len(df), freq="H")
    df["Date"] = df["Timestamp"].dt.date
    df_exploded = df.explode("Parsed Categories")
    df_cat = df_exploded[df_exploded["Parsed Categories"] == category]
    daily_counts = df_cat.groupby("Date").size()
    fig, ax = plt.subplots()
    ax.plot(daily_counts.index, daily_counts.values, marker="o")
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    ax.set_title(f"Daily Frequency of {category}")
    plt.xticks(rotation=45)
    if save_path:
        fig.savefig(save_path)
    st.pyplot(fig)

def extract_sentiment(classification_text):
    """Extracts overall sentiment (Positive, Neutral, Negative) from the classification text."""
    text = classification_text.lower()
    if "positive" in text:
        return "Positive"
    elif "negative" in text:
        return "Negative"
    elif "neutral" in text:
        return "Neutral"
    return "Unknown"

def plot_sentiment_distribution(df, save_path=None):
    """Plots a bar chart of sentiment distribution across messages."""
    df["Sentiment"] = df["Classification"].apply(extract_sentiment)
    sentiment_counts = df["Sentiment"].value_counts()
    fig, ax = plt.subplots()
    ax.bar(sentiment_counts.index, sentiment_counts.values, color=["green", "blue", "red"])
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    ax.set_title("Sentiment Distribution")
    if save_path:
        fig.savefig(save_path)
    st.pyplot(fig)

def calculate_personality_score_by_sender(df):
    """
    Calculates a dummy personality score for each sender based on the proportion of manipulative messages.
    Higher score means less manipulative.
    """
    df_exploded = df.explode("Parsed Categories")
    scores = {}
    for sender in df_exploded["Sender"].unique():
        sender_df = df_exploded[df_exploded["Sender"] == sender]
        total = len(sender_df)
        manip_count = sender_df[sender_df["Parsed Categories"].isin(["Gaslighting", "Coercive Control"])].shape[0]
        score = 100 - (manip_count / total * 100) if total > 0 else 100
        scores[sender] = round(score, 2)
    return scores

def plot_personality_scores(scores, save_path=None):
    """Plots a bar chart of personality scores by sender."""
    senders = list(scores.keys())
    values = list(scores.values())
    fig, ax = plt.subplots()
    ax.bar(senders, values, color="purple")
    ax.set_xlabel("Sender")
    ax.set_ylabel("Personality Likelihood Score")
    ax.set_title("Personality Likelihood Score by Sender (Higher means less manipulative)")
    if save_path:
        fig.savefig(save_path)
    st.pyplot(fig)

def plot_category_distribution_pie_by_sender(df, sender, save_path=None):
    """
    For a given sender, plots a pie chart of category distribution.
    """
    df_exploded = df.explode("Parsed Categories")
    sender_df = df_exploded[df_exploded["Sender"].str.lower() == sender.lower()]
    cat_counts = sender_df["Parsed Categories"].value_counts()
    if cat_counts.empty:
        st.info(f"No categories found for {sender}.")
        return
    fig, ax = plt.subplots()
    ax.pie(cat_counts.values, labels=cat_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.set_title(f"Category Distribution for {sender}")
    if save_path:
        fig.savefig(save_path)
    st.pyplot(fig)

def plot_category_comparison(df, category, save_path=None):
    """
    Plots a bar chart comparing the number of messages for a specific category across senders.
    """
    df_exploded = df.explode("Parsed Categories")
    df_cat = df_exploded[df_exploded["Parsed Categories"] == category]
    counts = df_cat["Sender"].value_counts()
    fig, ax = plt.subplots()
    ax.bar(counts.index, counts.values, color=["blue", "orange"])
    ax.set_xlabel("Sender")
    ax.set_ylabel("Count")
    ax.set_title(f"Comparison of '{category}' Messages by Sender")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.5, f"{v} ({(v/counts.sum()*100):.1f}%)", ha='center', va='bottom')
    if save_path:
        fig.savefig(save_path)
    st.pyplot(fig)

################################
# 7. SENDER-SPECIFIC REPORT FUNCTIONS
################################

def generate_sender_report(df, sender):
    """
    Generates an aggregated summary and final report for a specific sender.
    Returns summary and report texts.
    """
    sender_df = df[df["Sender"].str.lower() == sender.lower()]
    summary = f"Report for {sender}\n\n" + generate_aggregated_summary(sender_df)
    report = generate_report(summary)
    return summary, report

def save_report_to_file(report_text, filename):
    """
    Saves report text to a file in the results folder.
    """
    filepath = os.path.join(RESULTS_FOLDER, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_text)
    st.success(f"Report saved to {filepath}")

################################
# 8. STREAMLIT DASHBOARD
################################

def run_dashboard():
    st.title("Communication Analytics Platform")
    
    st.sidebar.header("Step 1: Upload PDF")
    uploaded_pdf = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded_pdf is None:
        st.sidebar.info("Please upload a PDF file to begin.")
        st.stop()
    
    # Step 2: Data Extraction
    if st.sidebar.button("Extract Data"):
        with st.spinner("Extracting messages from PDF..."):
            df_extracted = pd.DataFrame(process_pdf_file(uploaded_pdf))
            st.session_state.df_extracted = df_extracted
        st.success("Data extraction complete!")
        st.write("Extracted Data:")
        st.dataframe(st.session_state.df_extracted)
        # Save extraction to Excel
        extraction_excel = os.path.join(RESULTS_FOLDER, "ExtractedMessages.xlsx")
        st.session_state.df_extracted.to_excel(extraction_excel, index=False)
        st.success(f"Extracted data saved to {extraction_excel}")
    
    if "df_extracted" not in st.session_state:
        st.warning("Please extract data first using the sidebar button.")
        st.stop()
    
    # Step 3: Groq Classification
    if st.sidebar.button("Run Groq Classification"):
        with st.spinner("Classifying messages..."):
            df_classified = run_classification(st.session_state.df_extracted)
            st.session_state.df_classified = df_classified
        st.success("Classification complete!")
        st.write("Classified Data:")
        st.dataframe(st.session_state.df_classified)
        # Save classified data to Excel
        classified_excel = os.path.join(RESULTS_FOLDER, "ClassifiedMessages.xlsx")
        st.session_state.df_classified.to_excel(classified_excel, index=False)
        st.success(f"Classified data saved to {classified_excel}")
    
    if "df_classified" not in st.session_state:
        st.warning("Please run classification first.")
        st.stop()
    
    df = st.session_state.df_classified.copy()
    
    # Keyword Filter
    st.sidebar.header("Keyword Filter")
    keyword_input = st.sidebar.text_input("Enter keyword(s) (comma separated)", "")
    if keyword_input:
        keywords = [kw.strip().lower() for kw in keyword_input.split(",") if kw.strip()]
        def contains_keywords(text):
            return any(kw in text.lower() for kw in keywords)
        df_keyword = df[df["Message Text"].apply(contains_keywords)]
        st.markdown("### Messages Containing Specified Keywords")
        st.dataframe(df_keyword[["Message Number", "Sent Time", "Sender", "Receiver", "Subject", "Message Text"]])
    
    # Step 4: Separate messages by sender
    parent_a_msgs, parent_b_msgs = separate_by_sender(df.to_dict("records"))
    df_a = pd.DataFrame(parent_a_msgs)
    df_b = pd.DataFrame(parent_b_msgs)
    
    st.markdown("## Sender-Specific Data")
    st.write("Messages from **Barbara Anjola**:")
    st.dataframe(df_a)
    st.write("Messages from **Shola Anjola**:")
    st.dataframe(df_b)
    
    # Step 5: Visualizations
    st.markdown("## Detailed Visualizations")
    
    # Plot 1: Timeline Chart (Messages per Day)
    st.subheader("Timeline: Messages per Day")
    plot_messages_per_day(df, save_path=os.path.join(RESULTS_FOLDER, "MessagesPerDay.png"))
    
    # Plot 2: Stacked Category Distribution by Sender
    st.subheader("Stacked Category Distribution by Sender")
    plot_category_stacked_by_sender(df, save_path=os.path.join(RESULTS_FOLDER, "StackedCategoryBySender.png"))
    
    # Plot 3: Specific Category Frequency Over Time
    st.subheader("Daily Frequency: Gaslighting")
    plot_specific_categories_over_time(df, "Gaslighting", save_path=os.path.join(RESULTS_FOLDER, "GaslightingOverTime.png"))
    st.subheader("Daily Frequency: Coercive Control")
    plot_specific_categories_over_time(df, "Coercive Control", save_path=os.path.join(RESULTS_FOLDER, "CoerciveControlOverTime.png"))
    
    # Plot 4: Sentiment Distribution
    st.subheader("Sentiment Distribution")
    plot_sentiment_distribution(df, save_path=os.path.join(RESULTS_FOLDER, "SentimentDistribution.png"))
    
    # Plot 5: Personality Scores
    st.subheader("Personality Likelihood Score by Sender")
    scores = calculate_personality_score_by_sender(df)
    plot_personality_scores(scores, save_path=os.path.join(RESULTS_FOLDER, "PersonalityScores.png"))
    
    # Plot 6: Pie Chart for Category Distribution by Sender (separate for each sender)
    st.markdown("### Category Distribution by Sender (Pie Charts)")
    for sender in df["Sender"].unique():
        st.subheader(f"Category Distribution for {sender}")
        plot_category_distribution_pie_by_sender(df, sender, save_path=os.path.join(RESULTS_FOLDER, f"CategoryPie_{sender.replace(' ', '_')}.png"))
    
    # New: For each category, a comparison bar chart across senders
    st.markdown("### Category Comparison Across Senders")
    category_list = [
        "Coercive Control", "Gaslighting", "False Accusations",
        "Emotional Manipulation", "Guilt-Tripping",
        "Self-Advocating", "Cooperative", "Emotional Overwhelm"
    ]
    for cat in category_list:
        st.subheader(f"Comparison for {cat}")
        plot_category_comparison(df, cat, save_path=os.path.join(RESULTS_FOLDER, f"CategoryComparison_{cat.replace(' ', '_')}.png"))
    
    # Step 6: Aggregated Summary & Final Reports (Separate per Sender)
    st.markdown("## Sender-Specific Aggregated Reports")
    for sender in df["Sender"].unique():
        st.subheader(f"Final Report for {sender}")
        sender_df = df[df["Sender"].str.lower() == sender.lower()]
        summary_text = generate_aggregated_summary(sender_df)
        report_text = generate_report(summary_text)
        st.text_area(f"Report for {sender}", report_text, height=300)
        report_filename = f"FinalReport_{sender.replace(' ', '_')}.txt"
        save_report_to_file(report_text, report_filename)
    
    # Step 7: Save Classified Data to Excel
    if st.sidebar.button("Save Classified Data to Excel (Final)"):
        excel_path = os.path.join(RESULTS_FOLDER, "Final_ClassifiedMessages.xlsx")
        df.to_excel(excel_path, index=False)
        st.success(f"Data saved to {excel_path}.")

if __name__ == "__main__":
    run_dashboard()
