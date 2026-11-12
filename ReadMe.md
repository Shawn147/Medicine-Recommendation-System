# Medicine Recommendation System

A Streamlit assistant for educational health information, symptom follow-ups and
medical report explanations. Condition matches and medicine references are not
clinically validated diagnoses or prescriptions.

## Requirements

Python 3.9, 3.10, 3.11, or 3.12. Python 3.13 and newer has no wheels for these
pins, so pip must not be pointed at that interpreter.

| Package               | Version                             |
| --------------------- | ----------------------------------- |
| streamlit             | 1.41.1                              |
| pyarrow               | 19.0.1                              |
| pandas                | 2.2.3                               |
| numpy                 | 1.26.4                              |
| scikit-learn          | 1.5.2                               |
| scipy                 | 1.17.1 on Python 3.11–3.12; 1.13.1 on 3.9–3.10 |
| torch                 | 2.8.0+cpu on Linux, 2.8.0 otherwise |
| transformers          | 4.57.6                              |
| safetensors           | 0.7.0                               |
| sentence-transformers | 5.1.2                               |
| pypdf                 | 6.1.1                               |

`packages.txt` installs `tesseract-ocr` on Streamlit Community Cloud. NumPy,
pandas, PyArrow, scikit-learn, SciPy, and PyTorch are installed from wheels only.

## Run locally

Python 3.12 is already enough. From `shahan_mrs`:

```bash
bash run.sh
```

On Windows, double-click `run.bat`, or run it from this folder. The script
installs the packages and opens the app. It does not use the `streamlit`
command on the system path.

Before the app opens, the same script checks that SciPy imported completely
and that the saved symptom model matches the JSON data. A missing or outdated
model is downloaded and trained locally first. That step needs network access
and takes several minutes. Later runs skip it.

The saved symptom models run locally. General answers require local Ollama or a
configured hosted provider; see [setup and deployment](docs/implementation.md).
Keep API keys in environment variables or `.streamlit/secrets.toml`, never in Git.

## Features

- Symptom conversations with follow-up questions and emergency handling.
- Medical report uploads with PDF text extraction and photo OCR.
- Editable voice transcripts, prompt editing and browser conversation history.
- Source-linked health information and location-based hospital search links.
- Anonymous usage analytics, dataset previews and model evaluation.

## Project structure

```text
app.py                  Streamlit entry point
src/
  data/                 Load and validate prepared data
  ml/                   Features, training, evaluation and prediction
  chatbot/              Dialogue, follow-ups, safety and general answers
  services/             Medical report extraction and analytics storage
  browser/              Browser history, voice and file-drop bridges
  ui/                   Streamlit interface
  paths.py              Shared project paths
assets/                 Styles, logo, browser components and project images
data/                   Prepared datasets, reference records and provenance
artifacts/              Saved models, evaluation results and local analytics
scripts/                Data collection, preparation, audits and training
tests/                 Python and browser-component regression tests
notebooks/              Original exploratory notebook
docs/                   Setup, architecture, background and project documents
  archive/              Original prototype and course specification
  deliverables/         Project report and presentation
references/             Original source datasets and course materials
output/                 Synthetic sample reports and verification evidence
```

Start with [the teacher walkthrough](docs/teacher-guide.md) for a file-by-file presentation of the ML workflow and interface.
See [the code guide](docs/code-guide.md) for where to make changes and
[implementation notes](docs/implementation.md) for datasets, model limitations,
training, provider settings and deployment. The original course overview and
literature review are preserved in [project background](docs/project-background.md).

## Verify changes

Run from the `shahan_mrs` folder:

```bash
python scripts/run_tests.py
# Optional real local Qwen + classifier + web smoke tests:
python scripts/smoke_chat.py
node tests/test_chat_scroll.cjs
node tests/test_drop_component.cjs
node tests/test_voice_component.cjs
```

The Python suite needs the installed dependencies and saved model artifacts.
The browser checks simulate component events without using a real microphone.

## Data and privacy

Conversation history and readable report text are stored in the visitor's browser.
Submitted questions and readable reports may be sent to the configured AI provider.
Anonymous aggregate analytics are stored on the app server. The analytics dashboard
has no authentication; see the implementation notes before sharing a deployment.

For learning, research and prototyping only. See [LICENSE](LICENSE).
