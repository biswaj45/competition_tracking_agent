# Competition Tracking Agent

A modular Python agent for tracking competitors in the fraud analytics industry. The agent collects data on competitor product releases, pricing, and announcements, and provides analytics via CLI and dashboard.

## Features
- Modular data collection (web scraping, APIs)
- SQLite-based storage
- CLI reporting (using Rich)
- Dashboard (using Streamlit)

## Project Structure
- `competition_agent/` - Main package
  - `data_collection/` - Data collection modules
  - `storage/` - Data storage and access
  - `reporting/` - CLI reporting
  - `dashboard/` - Dashboard interface
- `main.py` - Entry point
- `requirements.txt` - Dependencies

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Run CLI: `python main.py`
3. Run dashboard: `streamlit run competition_agent/dashboard/app.py`
