#!/bin/bash

# Download spaCy model at container startup
python -m spacy download en_core_web_sm

# Launch Streamlit app
streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false