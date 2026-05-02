import argparse
import sys
import os
import json
import pandas as pd
import numpy as np
import re
from collections import Counter
from itertools import pairwise
from wordcloud import STOPWORDS
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Optional: Import these if you uncomment Phase 2 (Topic Modeling)
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.decomposition import LatentDirichletAllocation

# Import shared logic
import text_processor as tp

def setup_args():
    parser = argparse.ArgumentParser(description="Harvester: Large Scale Text Processor")
    parser.add_argument("--input", required=True, help="Path to input file (CSV/Excel) or folder")
    parser.add_argument("--col", required=True, help="Name of the text column to analyze")
    parser.add_argument("--output", default="sketch.json", help="Output path for the sketch file (.json)")
    parser.add_argument("--chunksize", type=int, default=50000, help="Rows per chunk")
    # parser.add_argument("--topics", type=int, default=5, help="Number of LDA topics (Disabled for speed)")
    parser.add_argument("--no-bigrams", action="store_true", help="Disable bigram calculation")
    return parser.parse_args()

def stream_file(file_path, chunksize, text_col):
    """Yields lists of raw text strings from CSV or Excel – now truly streaming for Excel."""
    ext = file_path.lower().split(".")[-1]

    if ext == 'csv':
        for chunk in pd.read_csv(file_path, usecols=[text_col], chunksize=chunksize, 
                                 on_bad_lines='skip', encoding='utf-8', engine='python'):
            yield chunk[text_col].dropna().astype(str).tolist()
        
    elif ext in ['xlsx', 'xlsm']:
        import openpyxl
        print(f"Loading Excel file {file_path} (streaming)...")
        wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        ws = wb.active
        rows = []
        count = 0
        col_idx = None
        
        for row_idx, row in enumerate(ws.iter_rows(values_only=True), start=1):
            if row_idx == 1:  # header row
                header = [str(cell) if cell else "" for cell in row]
                try:
                    col_idx = header.index(text_col)
                except ValueError:
                    try:
                        col_idx = [c.lower() for c in header].index(text_col.lower())
                    except ValueError:
                        print(f"Column '{text_col}' not found. Using first column.")
                        col_idx = 0
                continue
            
            if col_idx is not None:
                cell_val = row[col_idx] if col_idx < len(row) else None
                if cell_val is not None:
                    text = str(cell_val).strip()
                    if text and text.lower() not in ['nan', 'none', 'null']:
                        rows.append(text)
                        count += 1
            
            if len(rows) >= chunksize:
                yield rows
                rows = []
        
        if rows:
            yield rows
        wb.close()

def main():
    args = setup_args()
    
    # 1. Setup cleaning resources
    try: nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError: nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    
    stopwords = set(STOPWORDS)
    stopwords.update(tp.default_prepositions())
    trans_map = tp.build_punct_translation(keep_hyphens=False, keep_apostrophes=False)
    
    # 2. State Aggregators
    word_counter = Counter()
    bigram_counter = Counter()
    total_docs = 0
    
    pos_thresh = 0.05
    neg_thresh = -0.05
    
    print(f"--- HARVESTER STARTED ---")
    print(f"Input: {args.input} | Column: {args.col}")
    
    # --- PASS 1: VOCABULARY & COUNTERS ---
    print(">>> Phase 1: Building Vocabulary and Counters...")
    
    files = [args.input] if os.path.isfile(args.input) else [os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith(('csv', 'xlsx'))]
    
    for fpath in files:
        for chunk_idx, texts in enumerate(stream_file(fpath, args.chunksize, args.col)):
            
            for text in texts:
                tokens = tp.clean_and_tokenize(
                    text, True, True, True, True, trans_map, stopwords, None, 2, True
                )
                if not tokens: continue
                
                # Update Counters
                word_counter.update(tokens)
                if not args.no_bigrams and len(tokens) > 1:
                    bigram_counter.update(tuple(pairwise(tokens)))
            
            total_docs += len(texts)
            print(f"   Processed chunk {chunk_idx+1} (approx {total_docs} rows)...")

    print(f"Phase 1 Complete. Vocab Size: {len(word_counter)}")
    
    # --- PASS 2: TOPIC MODELING (ONLINE LDA) ---
    # NOTE: Disabled for efficiency. 
    # The Main App generates topics dynamically from "topic_docs". 
    # Since Harvester does not save raw document vectors (to keep JSON small and private),
    # there is no need to spend CPU time calculating LDA models here that won't be displayed.
    
    # print(">>> Phase 2: Training Online LDA (Skipped for optimization)...")
    # ... (LDA code removed to save I/O and CPU) ...

    # -sentimnt aggregation
    print(">>> Finalizing Sentiment Stats...")
    # to calculate sentiment based on unique vocabulary (much faster than row-by-row)
    vocab_sents = {w: sia.polarity_scores(w)['compound'] for w in word_counter.keys()}
    
    # not explicitly saved in JSON fields, but useful if for possibly extending functionality later
    # final_pos_count = sum(word_counter[w] for w, s in vocab_sents.items() if s >= pos_thresh)
    # final_neg_count = sum(word_counter[w] for w, s in vocab_sents.items() if s <= neg_thresh)
    # sentiment aggregation intentionally skipped tho — the main app computes it on demand

    # -serialization
    
    # flatten bi-grams for JSON: {('a','b'): 5} -> {'a|b': 5}
    serializable_bigrams = {f"{k[0]}|{k[1]}": v for k, v in bigram_counter.items()}
    
    # schema construction
    sketch_data = {
        "total_rows": total_docs,
        "counts": dict(word_counter),
        "bigrams": serializable_bigrams,
        "topic_docs": [], # Empty for privacy/size; Main App Topic Modeling will be disabled for this file.
        "limit_reached": True,
        "temporal_counts": {}, 
        "entity_counts": {},   
        "doc_freqs": dict(word_counter), # Approx for TF-IDF
        "metadata": {"source": args.input, "col": args.col, "generated_by": "Harvester v2.9"}
    }
    
    # save
    out_path = args.output
    if not out_path.lower().endswith('.json'):
        out_path += '.json'

    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(sketch_data, f)
            
        print(f"✅ Sketch saved to: {out_path}")
        print(f"   Total Rows: {total_docs}")
        print(f"   Unique Words: {len(word_counter)}")
        print("   -> Upload this JSON file to the 'Offline Analysis' section in the Main App.")
    except Exception as e:
        print(f"❌ Error saving file: {e}")

if __name__ == "__main__":
    main()
