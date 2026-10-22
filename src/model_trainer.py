import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from src.preprocessor import clean_text

MODEL_REGISTRY = {
    "Logistic Regression": {
        "class": LogisticRegression,
        "default_params": {"C": 1.5, "max_iter": 1000, "random_state": 42},
        "has_proba": True
    },
    "Multinomial Naive Bayes": {
        "class": MultinomialNB,
        "default_params": {"alpha": 1.0},
        "has_proba": True
    },
    "Complement Naive Bayes": {
        "class": ComplementNB,
        "default_params": {"alpha": 1.0},
        "has_proba": True
    },
    "Linear SVM": {
        "class": LinearSVC,
        "default_params": {"C": 1.0, "random_state": 42, "dual": False},
        "has_proba": False
    },
    "Random Forest": {
        "class": RandomForestClassifier,
        "default_params": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
        "has_proba": True
    }
}

def train_and_evaluate_model(
    train_df,
    model_name="Logistic Regression",
    test_size=0.2,
    random_state=42,
    ngram_range=(1, 2),
    max_features=5000,
    model_kwargs=None
):
    """
    Trains a classification model using TF-IDF vectorization strictly fitted
    on the training split to prevent any data leakage.
    Returns model, vectorizer, validation data, and comprehensive evaluation metrics.
    """
    # 1. Clean text if not already cleaned
    if 'cleaned_text' not in train_df.columns:
        train_df = train_df.copy()
        train_df['cleaned_text'] = train_df['text'].apply(clean_text)
        
    # 2. Stratified Train-Validation Split (Ensures class balance is preserved)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        train_df['cleaned_text'],
        train_df['target'],
        test_size=test_size,
        random_state=random_state,
        stratify=train_df['target']
    )
    
    # 3. TF-IDF Vectorizer (Fit strictly on X_train to prevent Data Leakage)
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        stop_words='english',
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train_raw)
    X_val_vec = vectorizer.transform(X_val_raw)
    
    # 4. Instantiate Model
    config = MODEL_REGISTRY.get(model_name, MODEL_REGISTRY["Logistic Regression"])
    params = config["default_params"].copy()
    if model_kwargs:
        params.update(model_kwargs)
        
    model = config["class"](**params)
    model.fit(X_train_vec, y_train)
    
    # 5. Validation Predictions
    y_pred = model.predict(X_val_vec)
    
    # Probabilities / Decision scores for ROC
    y_scores = None
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_val_vec)[:, 1]
    elif hasattr(model, "decision_function"):
        dec = model.decision_function(X_val_vec)
        # Apply sigmoid to normalize to 0-1
        y_scores = 1 / (1 + np.exp(-dec))
        
    # 6. Evaluation Metrics calculation (100% genuine validation calculations)
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_val, y_pred)
    report_dict = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_val, y_pred, target_names=['Non-Disaster (0)', 'Disaster (1)'])
    
    fpr, tpr, roc_auc = None, None, 0.5
    if y_scores is not None:
        fpr, tpr, _ = roc_curve(y_val, y_scores)
        roc_auc = auc(fpr, tpr)
        
    # 7. Extract Top Informative Features (if model provides coefficients or log probabilities)
    top_disaster_words = []
    top_nondisaster_words = []
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    if hasattr(model, 'coef_'):
        coefs = model.coef_[0]
        top_d_idx = np.argsort(coefs)[-15:][::-1]
        top_nd_idx = np.argsort(coefs)[:15]
        top_disaster_words = [(feature_names[i], float(coefs[i])) for i in top_d_idx]
        top_nondisaster_words = [(feature_names[i], float(coefs[i])) for i in top_nd_idx]
    elif hasattr(model, 'feature_log_prob_'):
        # For Naive Bayes, log prob difference
        log_prob_diff = model.feature_log_prob_[1] - model.feature_log_prob_[0]
        top_d_idx = np.argsort(log_prob_diff)[-15:][::-1]
        top_nd_idx = np.argsort(log_prob_diff)[:15]
        top_disaster_words = [(feature_names[i], float(log_prob_diff[i])) for i in top_d_idx]
        top_nondisaster_words = [(feature_names[i], float(log_prob_diff[i])) for i in top_nd_idx]
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[-15:][::-1]
        top_disaster_words = [(feature_names[i], float(importances[i])) for i in top_idx]
        top_nondisaster_words = []
        
    results = {
        "model_name": model_name,
        "model": model,
        "vectorizer": vectorizer,
        "train_size": len(X_train_raw),
        "val_size": len(X_val_raw),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "confusion_matrix": cm,
        "classification_report_dict": report_dict,
        "classification_report_text": report_text,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": float(roc_auc),
        "top_disaster_words": top_disaster_words,
        "top_nondisaster_words": top_nondisaster_words,
        "val_df": pd.DataFrame({
            'text': train_df.loc[X_val_raw.index, 'text'],
            'actual': y_val,
            'predicted': y_pred,
            'confidence': y_scores if y_scores is not None else [1.0 if p == 1 else 0.0 for p in y_pred]
        })
    }
    
    return results

def benchmark_all_models(train_df, test_size=0.2, random_state=42):
    """
    Runs an actual benchmark across all supported algorithms
    on the exact same train/validation split.
    """
    results_list = []
    
    for name in MODEL_REGISTRY.keys():
        res = train_and_evaluate_model(
            train_df=train_df,
            model_name=name,
            test_size=test_size,
            random_state=random_state
        )
        results_list.append({
            "Algorithm": name,
            "Accuracy": res["accuracy"],
            "Precision": res["precision"],
            "Recall": res["recall"],
            "F1-Score": res["f1_score"],
            "ROC-AUC": res["roc_auc"]
        })
        
    return pd.DataFrame(results_list)

def predict_single_tweet(model, vectorizer, raw_text):
    """
    Predicts whether a single tweet is disaster-related or not.
    Returns dict with cleaned text, prediction (0 or 1), label, confidence,
    and identified clue words present in the tweet.
    """
    cleaned = clean_text(raw_text)
    if not cleaned:
        return {
            "raw_text": raw_text,
            "cleaned_text": "",
            "target": 0,
            "label": "Non-Disaster-Related",
            "confidence": 0.50,
            "tokens": [],
            "contributing_words": []
        }
        
    vec_input = vectorizer.transform([cleaned])
    pred = int(model.predict(vec_input)[0])
    
    confidence = 0.50
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(vec_input)[0]
        confidence = float(probs[pred])
    elif hasattr(model, 'decision_function'):
        dec = float(model.decision_function(vec_input)[0])
        sig = 1 / (1 + np.exp(-abs(dec)))
        confidence = float(sig)
        
    # Analyze contributing tokens present in the tweet
    words = cleaned.split()
    feature_names = set(vectorizer.get_feature_names_out())
    present_features = [w for w in words if w in feature_names]
    
    token_weights = []
    if hasattr(model, 'coef_'):
        vocab = vectorizer.vocabulary_
        coefs = model.coef_[0]
        for w in present_features:
            if w in vocab:
                w_idx = vocab[w]
                w_weight = coefs[w_idx]
                token_weights.append({
                    "word": w,
                    "weight": float(w_weight),
                    "signals": "Disaster" if w_weight > 0 else "Non-Disaster"
                })
        token_weights.sort(key=lambda x: abs(x["weight"]), reverse=True)
        
    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned,
        "target": pred,
        "label": "Disaster-Related Tweet" if pred == 1 else "Non-Disaster Tweet",
        "confidence": confidence,
        "tokens": words,
        "contributing_words": token_weights
    }

def predict_test_dataset(model, vectorizer, test_df):
    """
    Runs batch predictions on test.csv without modifying test_df.
    Returns (predictions_df, submission_df).
    """
    test_copy = test_df.copy()
    if 'cleaned_text' not in test_copy.columns:
        test_copy['cleaned_text'] = test_copy['text'].apply(clean_text)
        
    X_test_vec = vectorizer.transform(test_copy['cleaned_text'])
    preds = model.predict(X_test_vec)
    
    confidences = []
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_test_vec)
        confidences = [float(probs[i][pred]) for i, pred in enumerate(preds)]
    elif hasattr(model, 'decision_function'):
        dec = model.decision_function(X_test_vec)
        confidences = [float(1 / (1 + np.exp(-abs(d)))) for d in dec]
    else:
        confidences = [1.0] * len(preds)
        
    test_copy['predicted_target'] = preds
    test_copy['predicted_label'] = test_copy['predicted_target'].map({
        0: 'Non-Disaster (0)',
        1: 'Disaster (1)'
    })
    test_copy['confidence'] = confidences
    
    # Kaggle submission format strictly matching sample_submission.csv
    submission_df = test_copy[['id', 'predicted_target']].rename(columns={'predicted_target': 'target'})
    
    return test_copy, submission_df
