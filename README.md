# econ-lab-23-fedspeak
Lab 23: FedSpeak 2.0 — NLP Pipeline, Embeddings &amp; Prediction

## Objective
Develop and validate a production-grade NLP pipeline for extracting sentiment signals from Federal Reserve meeting minutes, evaluating lexical (TF-IDF) versus semantic (sentence-transformer) representations for predicting monetary policy tightening cycles.

## Methodology
- **Diagnostic analysis:** Identified and corrected three systematic errors in a broken NLP pipeline — naive whitespace tokenization, domain-inappropriate sentiment dictionary (Harvard GI), and uninformative TF-IDF parameterization (`min_df=1`, `max_df=1.0`)
- **Preprocessing:** Implemented proper tokenization via `nltk.word_tokenize` with regex cleaning, stopword removal, and lemmatization; verified zero non-alphabetic token leakage
- **Sentiment scoring:** Replaced Harvard General Inquirer with the Loughran-McDonald (LM) financial dictionary, reducing false positive rate from ~56% to near zero for neutral financial terms (e.g., *capital*, *debt*, *liability*)
- **Feature engineering:** Corrected TF-IDF parameters (`min_df=5`, `max_df=0.85`, `ngram_range=(1,2)`) to eliminate background noise and capture economically meaningful bigrams such as *interest rate* and *price stability*
- **Semantic embeddings:** Encoded 240 FOMC minutes using `all-MiniLM-L6-v2` (384-dimensional dense vectors) via `sentence-transformers`
- **Clustering comparison:** Applied K-Means (K=3) to both representations; evaluated cluster quality via silhouette score and visualized in PCA-reduced 2D space
- **Predictive evaluation:** Used `TimeSeriesSplit` (5 folds) with logistic regression to assess AUC-ROC for predicting Fed tightening cycles on a binary label derived from historical rate-hiking periods

## Key Findings
TF-IDF achieved higher out-of-sample predictive accuracy (AUC = 0.82 ± 0.21) compared to sentence-transformer embeddings (AUC = 0.72 ± 0.21) for classifying Federal Reserve tightening cycles. While embeddings produced marginally better cluster separation (silhouette: 0.197 vs. 0.168), TF-IDF bigram features more directly capture the distinctive lexical patterns — *inflation elevated*, *raise rates*, *policy tightening* — that characterize hawkish FOMC communication. The elevated standard deviation across folds reflects structural breaks across monetary policy regimes and limited labeled observations, underscoring the importance of domain-specific feature engineering over general-purpose semantic models in central bank text analysis.

## Module
**`src/fomc_sentiment.py`** — reusable analysis module exposing:
- `preprocess_fomc(text)` — tokenization, cleaning, lemmatization
- `compute_lm_sentiment(text)` — Loughran-McDonald sentiment scoring with uncertainty index
- `build_tfidf_matrix(texts)` — configurable TF-IDF vectorization pipeline
