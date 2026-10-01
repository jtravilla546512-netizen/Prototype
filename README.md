# Media Bias Analysis — prototype

This Streamlit app is a **UI demonstration with seeded example outputs**. It does not load trained weights, call a model, or perform real inference. Labels and percentages are illustrative placeholders, not research results.

## Run it

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Choose one of the seeded examples or enter a sentence and optional article text. Click one or more model buttons to reveal the corresponding demonstration result:

- Baseline replication — sentence only
- Controlled baseline — sentence only, aligned to the enhanced pilot comparison
- Enhanced model — target sentence with article context

The sample examples have preset demonstration labels. Custom-text results use stable placeholder values so the presentation is repeatable. Neither is a prediction from the trained system. The interface can later be connected to model checkpoints when those are ready.
