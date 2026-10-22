import re
import html
import string
import pandas as pd
import numpy as np

# Standard English stopwords fallback
FALLBACK_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll",
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
    'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
    'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
    "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
    'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
    "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
    "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn',
    "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

def get_stopwords():
    """
    Returns stopwords from nltk if available, else fallback set.
    """
    try:
        import nltk
        from nltk.corpus import stopwords
        return set(stopwords.words('english'))
    except Exception:
        return FALLBACK_STOPWORDS

def clean_text(text, remove_urls=True, remove_mentions=True, remove_hashtags_symbol=True,
               remove_html=True, remove_punct=True, lowercase=True, remove_numbers=False):
    """
    Standard comprehensive cleaning function for tweets.
    """
    if pd.isna(text) or not str(text).strip():
        return ""
    
    text = str(text)
    
    # 1. Decode HTML entities (&amp; -> &, &lt; -> <, etc.)
    if remove_html:
        text = html.unescape(text)
        text = re.sub(r'<.*?>+', '', text)
        
    # 2. Convert to lowercase
    if lowercase:
        text = text.lower()
        
    # 3. Remove URLs
    if remove_urls:
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
    # 4. Remove User Mentions (@username)
    if remove_mentions:
        text = re.sub(r'@\w+', '', text)
        
    # 5. Handle Hashtags (#earthquake -> earthquake)
    if remove_hashtags_symbol:
        text = re.sub(r'#(\w+)', r'\1', text)
        
    # 6. Remove numbers if specified
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
        
    # 7. Remove Punctuation & Special Characters
    if remove_punct:
        # replace punctuation with space to prevent words from sticking together
        punct_pattern = re.compile('[%s]' % re.escape(string.punctuation))
        text = punct_pattern.sub(' ', text)
        
    # 8. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def clean_text_step_by_step(text):
    """
    Produces step-by-step intermediate transformations of a tweet
    for educational and interactive inspection.
    """
    original = str(text) if not pd.isna(text) else ""
    
    step1_html = html.unescape(original)
    step1_html = re.sub(r'<.*?>+', '', step1_html)
    
    step2_lower = step1_html.lower()
    
    step3_urls = re.sub(r'https?://\S+|www\.\S+', '', step2_lower)
    
    step4_mentions = re.sub(r'@\w+', '', step3_urls)
    
    step5_hashtags = re.sub(r'#(\w+)', r'\1', step4_mentions)
    
    punct_pattern = re.compile('[%s]' % re.escape(string.punctuation))
    step6_punct = punct_pattern.sub(' ', step5_hashtags)
    
    step7_final = re.sub(r'\s+', ' ', step6_punct).strip()
    
    # Tokenization & Stopwords removal
    tokens = step7_final.split()
    stops = get_stopwords()
    tokens_no_stop = [tok for tok in tokens if tok not in stops and len(tok) > 1]
    
    return {
        "1_Original": original,
        "2_HTML_Decoded": step1_html,
        "3_Lowercase": step2_lower,
        "4_No_URLs": step3_urls,
        "5_No_Mentions": step4_mentions,
        "6_Clean_Hashtags": step5_hashtags,
        "7_No_Punctuation": step6_punct,
        "8_Normalized_Final": step7_final,
        "9_Tokens": tokens,
        "10_Tokens_No_Stopwords": tokens_no_stop
    }

def extract_tweet_features(df, text_col='text'):
    """
    Extracts EDA text features such as character length, word count,
    average word length, presence of URLs, hashtags, mentions.
    Returns a copy of the dataframe with added feature columns.
    """
    df_feat = df.copy()
    
    df_feat['char_length'] = df_feat[text_col].astype(str).apply(len)
    df_feat['word_count'] = df_feat[text_col].astype(str).apply(lambda s: len(s.split()))
    df_feat['avg_word_length'] = df_feat.apply(
        lambda r: (r['char_length'] / r['word_count']) if r['word_count'] > 0 else 0, axis=1
    )
    df_feat['has_url'] = df_feat[text_col].astype(str).apply(
        lambda s: 1 if re.search(r'https?://\S+|www\.\S+', s) else 0
    )
    df_feat['has_mention'] = df_feat[text_col].astype(str).apply(
        lambda s: 1 if re.search(r'@\w+', s) else 0
    )
    df_feat['has_hashtag'] = df_feat[text_col].astype(str).apply(
        lambda s: 1 if re.search(r'#\w+', s) else 0
    )
    
    return df_feat

def create_cleaned_working_copy(df, fill_missing=True, drop_duplicates=False):
    """
    Creates an independent cleaned working copy of the dataset without modifying original.
    """
    working_df = df.copy()
    
    # Missing value handling
    if fill_missing:
        if 'keyword' in working_df.columns:
            working_df['keyword'] = working_df['keyword'].fillna('none')
        if 'location' in working_df.columns:
            working_df['location'] = working_df['location'].fillna('unknown')
            
    # Text cleaning
    if 'text' in working_df.columns:
        working_df['cleaned_text'] = working_df['text'].apply(clean_text)
        
    # Optional deduplication
    if drop_duplicates and 'text' in working_df.columns:
        working_df = working_df.drop_duplicates(subset=['text']).reset_index(drop=True)
        
    return working_df
