import streamlit as st
import random
import re

@st.cache_data
def load_classifier_data(file_path=".data/classifier.train.txt", max_examples=100):
    """Load classifier training data and parse examples"""
    examples = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by __label__ to get individual examples
    parts = re.split(r'__label__', content)[1:]  # Skip first empty part
    
    for part in parts:
        if not part.strip():
            continue
            
        lines = part.strip().split('\n', 1)
        if len(lines) >= 2:
            label = lines[0].strip()
            text = lines[1] if len(lines) > 1 else ""
        else:
            # Single line case
            first_space = part.find(' ')
            if first_space > 0:
                label = part[:first_space].strip()
                text = part[first_space+1:].strip()
            else:
                continue
                
        examples.append({
            'label': label,
            'text': text,
            'text_length': len(text),
            'word_count': len(text.split())
        })
        
        if len(examples) >= max_examples:
            break
    
    return examples

def main():
    st.set_page_config(page_title="Quality Classifier Data Viewer", layout="wide")
    st.title("Quality Classifier Training Data Viewer")
    st.write("View random examples from classifier training data")
    
    # Load data
    with st.spinner("Loading classifier data..."):
        examples = load_classifier_data()
    
    st.write(f"Loaded {len(examples)} examples")
    
    # Summary stats
    high_quality = sum(1 for ex in examples if ex['label'] == 'high_quality')
    low_quality = sum(1 for ex in examples if ex['label'] == 'low_quality')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("High Quality", high_quality)
    with col2:
        st.metric("Low Quality", low_quality)
    with col3:
        avg_length = sum(ex['text_length'] for ex in examples) / len(examples)
        st.metric("Avg Text Length", f"{avg_length:.0f}")
    
    # Filter controls
    st.subheader("Filter Examples")
    label_filter = st.selectbox("Filter by label:", ["All", "high_quality", "low_quality"])
    
    filtered_examples = examples
    if label_filter != "All":
        filtered_examples = [ex for ex in examples if ex['label'] == label_filter]
    
    # Random example button
    if st.button("Show Random Example") or len(filtered_examples) > 0:
        if filtered_examples:
            example = random.choice(filtered_examples)
            
            st.subheader(f"Random Example - Label: {example['label']}")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.text_area("Text Content", example['text'], height=400, key=f"text_{random.randint(1,10000)}")
            
            with col2:
                st.write("**Stats:**")
                st.write(f"Label: {example['label']}")
                st.write(f"Characters: {example['text_length']:,}")
                st.write(f"Words: {example['word_count']:,}")
                
                # Quality indicator
                if example['label'] == 'high_quality':
                    st.success(" High Quality")
                else:
                    st.error("L Low Quality")
    
    # Show multiple random examples
    st.subheader("Multiple Random Examples")
    num_examples = st.slider("Number of examples to show:", 1, 10, 3)
    
    if st.button("Show Multiple Examples"):
        random_examples = random.sample(filtered_examples, min(num_examples, len(filtered_examples)))
        
        for i, example in enumerate(random_examples):
            with st.expander(f"Example {i+1} - {example['label']} ({example['text_length']} chars)"):
                st.text_area(f"Content {i+1}", example['text'], height=200, key=f"multi_{i}")

if __name__ == "__main__":
    main()