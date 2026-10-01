# DA-RoBERTa Media Bias Analysis

Single-model Streamlit interface prototype for a defense presentation. Choose one of the seeded examples or enter a target sentence and its full article. The app animates article splitting, target matching, nearby-sentence selection, and the enhanced-model analysis view.

The displayed result is illustrative and is not computed from trained DA-RoBERTa weights. It must not be reported as an experimental prediction or result.

## Seeded demonstration inputs

### Biased example

Target sentence: “The mayor’s reckless decision devastated the community, critics said.”

Article: “Residents gathered outside city hall on Tuesday to discuss the new budget. The mayor’s reckless decision devastated the community, critics said. The mayor’s office said the changes were intended to reduce costs. City council members are expected to review the plan next month.”

### Non-biased example

Target sentence: “The city council approved the revised transportation budget on Tuesday.”

Article: “City council members met on Tuesday to review the proposed budget. The city council approved the revised transportation budget on Tuesday. The plan allocates funds for road maintenance and public transit. Officials said implementation will begin next month.”

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```
