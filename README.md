# HVAC Cash Flow Model v2

Streamlit-based HVAC planning tool for monthly cash flow forecasting, scenario analysis, and operational driver modeling.

## Key Features
- Schema v2 assumptions with migration support for legacy scenario JSON.
- Discrete monthly staffing events for technicians and sales staff (hires + attrition).
- Segmented residential and light-commercial maintenance, upsell, and new-build revenue engines.
- Goal-seek solver for scalar inputs with bounded search.
- Nominal, inflation-adjusted real, and discounted PV real value presentation.
- Local scenario/workspace save-load, plus JSON import/export bundles.
- User-selectable sensitivity drivers and tornado analysis.
- Range-based analytics and configurable chart/table views.

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

## Run Tests
```powershell
.\.venv\Scripts\python -m pytest -q
```

## Build Clickable Windows App
```powershell
.\build_windows_exe.ps1
```

After build:
- Run: `dist\HVAC_Cashflow\HVAC_Cashflow.exe`
- Share by zipping the whole `dist\HVAC_Cashflow` folder and sending it.

## Publish To Streamlit Community Cloud
1. Push this repo (or publish branch) to GitHub.
2. In Streamlit Community Cloud, create a new app pointing to:
   - Repo: your fork/repo
   - Branch: `publish/streamlit-community-cloud` (or `main`)
   - Main file path: `app.py`
3. Deploy. Streamlit installs from `requirements.txt`.

### Cloud Storage Behavior
- Scenarios, workspaces, runtime logs, and change requests use a configurable storage root.
- On launch, use the sidebar **Storage Location** block:
  - `App default storage`
  - `Session temp storage (cloud-friendly)` (recommended in multi-user cloud runtime)
  - `Custom server folder path`
- Browser local folders cannot be mounted directly in Streamlit Cloud.
- You can preconfigure server-side storage root with env var:
  - `HVAC_STORAGE_ROOT=/mount/data/hvac_cashflow`

### Persistence Note For Community Cloud
- Local container storage may reset across redeploy/restart.
- For durable history, periodically export scenarios/workspaces and/or back up the configured storage path.
