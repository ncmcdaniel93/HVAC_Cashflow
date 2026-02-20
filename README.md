# HVAC Cash Flow Model

## Setup

### PowerShell (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

### Bash

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app models a single HVAC business scenario using scalar drivers and creates a monthly cash flow forecast.
