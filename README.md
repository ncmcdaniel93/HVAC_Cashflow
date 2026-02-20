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
