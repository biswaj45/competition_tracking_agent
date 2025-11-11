# 🚀 Competition Tracking Agent - Complete Deployment Guide

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [System Requirements](#system-requirements)
3. [Complete Setup Instructions](#complete-setup-instructions)
4. [Project Structure](#project-structure)
5. [Configuration Details](#configuration-details)
6. [Execution Instructions](#execution-instructions)
7. [Key Features & Capabilities](#key-features--capabilities)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

**Competition Tracking Agent** is an AI-powered competitive intelligence system designed for **TransUnion's Fraud Analytics division**. It automatically scrapes, analyzes, and reports on 27+ competitors in the fraud detection, identity verification, and biometrics industry.

### Business Context
- **Industry**: Fraud Analytics, Identity Verification, Anti-Money Laundering (AML)
- **Target Competitors**: 27+ companies including Experian, Equifax, LexisNexis, FICO, Jumio, Onfido, BioCatch, etc.
- **Data Sources**: 12 RSS news feeds (TechCrunch, PYMNTS, Biometric Update, InfoSecurity Magazine, etc.)
- **Output**: Professional PDF reports with charts, JSON data files, and deduplicated article archives

### Core Capabilities
✅ **Async RSS Feed Scraping** - Parallel scraping of 12 news sources (80-second completion)
✅ **AI-Powered Analysis** - Hugging Face Transformers (T5, DistilBERT, BART) for summarization and classification
✅ **Intelligent Filtering** - Removes political articles, hacker news, stock listings, ads, and irrelevant content
✅ **Deduplication** - 80% similarity threshold using SequenceMatcher + Jaccard similarity
✅ **Competitor Tracking** - Mentions, sentiment analysis, and product/partnership detection
✅ **Fraud Domain Analysis** - Account Takeover, AML, Biometrics, Synthetic Identity, etc.
✅ **Technology Trends** - Fraud-specific keywords (biometric, facial recognition, liveness detection, behavioral analytics)
✅ **Professional PDF Reports** - Navy/blue theme with 4 charts (competitor bar, threats pie, tech trends, fraud domains)
✅ **Citation Tracking** - Top 10 key articles with clickable links

---

## 💻 System Requirements

### Minimum Requirements
```
Operating System: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
Python Version: 3.8.5 or higher (recommended: 3.8.x - 3.10.x)
RAM: 8 GB minimum (16 GB recommended for AI model loading)
Disk Space: 5 GB free space (for models, reports, and article cache)
Internet: Stable connection for RSS scraping and model downloads
```

### Python Environment
- **Python 3.8.5+** (tested version)
- **pip** 21.0 or higher
- **virtualenv** or **venv** (recommended for isolation)

### Browser (Optional)
- Chrome/Edge/Firefox for viewing generated PDF reports

---

## 🔧 Complete Setup Instructions

### Step 1: Clone or Transfer Project Files

#### Option A: If Using Git
```bash
git clone <repository_url>
cd competition_tracking_agent
```

#### Option B: Manual Transfer
1. Copy the entire project folder to your system
2. Ensure directory structure matches:
```
competition_tracking_agent/
├── competition_agent/          # Main package
│   ├── __init__.py
│   ├── config/                 # Configuration files
│   ├── data_collection/        # Web scraping modules
│   ├── storage/                # SQLite database models
│   ├── reporting/              # CLI reporting
│   ├── dashboard/              # Streamlit dashboard
│   ├── llm/                    # LLM integration
│   └── news/                   # News collection
├── tools/                      # Analysis and generation scripts
│   ├── improved_ci_analyzer.py         # Main analyzer
│   └── improved_pdf_generator.py       # PDF report generator
├── reports/                    # Output directory (auto-created)
├── article_cache/              # Cached articles (auto-created)
├── main.py                     # Entry point
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── COMPREHENSIVE_SYSTEM_GUIDE.md
├── QUICK_START_RESEARCH.md
└── PROJECT_STRUCTURE.md
```

### Step 2: Create Virtual Environment (Recommended)

#### Windows (PowerShell)
```powershell
# Navigate to project directory
cd competition_tracking_agent

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Linux/macOS
```bash
# Navigate to project directory
cd competition_tracking_agent

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

#### Core Dependencies (from requirements.txt)
```
requests            # HTTP requests
beautifulsoup4      # HTML parsing
python-dotenv       # Environment variables
pandas              # Data manipulation
streamlit           # Dashboard (optional)
rich                # CLI formatting
reportlab           # PDF generation
wordcloud           # Word cloud visualization
matplotlib          # Charts and plots
seaborn             # Statistical visualization
python-dateutil     # Date parsing
feedparser          # RSS feed parsing
textblob            # Text processing
nltk                # Natural language toolkit
python-linkedin-v2  # LinkedIn API (optional)
```

#### Additional Dependencies (Auto-installed by tools)
```
aiohttp             # Async HTTP (used in scraper)
transformers        # Hugging Face models
torch               # PyTorch (for AI models)
sentencepiece       # Tokenization
```

### Step 4: Download Required AI Models (First Run Only)

The system uses Hugging Face models that auto-download on first execution:

```bash
# Test model download (optional - models download automatically on first run)
python -c "from transformers import pipeline; print('Models will download on first run')"
```

**Models Used:**
- `t5-small` (77MB) - Text summarization
- `distilbert-base-uncased-finetuned-sst-2-english` (268MB) - Sentiment analysis
- `facebook/bart-large-mnli` (1.63GB) - Zero-shot classification

**First run will take 5-10 minutes** to download models. Subsequent runs are instant.

### Step 5: Verify Installation

```bash
# Check Python version
python --version  # Should show 3.8.5 or higher

# Check installed packages
pip list | grep -E "(requests|beautifulsoup4|reportlab|matplotlib|feedparser|transformers)"

# Test import (should show no errors)
python -c "import requests, feedparser, reportlab, matplotlib, transformers; print('✅ All imports successful')"
```

---

## 📁 Project Structure

### Core Components

#### 1. **Scraping System** (`tools/improved_ci_analyzer.py`)
**Purpose**: Scrape 12 RSS news feeds, filter junk articles, deduplicate, and analyze competitors

**Key Features**:
- **Async scraping** with `aiohttp` (80-second completion for 30 days)
- **12 RSS Sources**:
  - Biometric Update (biometricupdate.com/feed)
  - PYMNTS (pymnts.com/feed)
  - TechCrunch (techcrunch.com/feed)
  - Security Intelligence (securityintelligence.com/feed)
  - InfoSecurity Magazine (infosecurity-magazine.com/rss)
  - Finextra (finextra.com/rss/headlines.xml)
  - The Financial Brand (thefinancialbrand.com/feed)
  - Bank Info Security (bankinfosecurity.com/rss-feeds)
  - Payment Week (paymentweek.com/feed)
  - Digital Transactions (digitaltransactions.net/feed)
  - Biometric Today (biometrictoday.com/feed)
  - Identity Week (identityweek.net/feed)

- **Junk Filtering**:
  ```python
  # Political keywords filter
  ['trump', 'biden', 'election', 'white house', 'congress', 'senate', 'governor']
  
  # Hacker/spyware filter
  ['pegasus', 'spyware', 'nso group', 'hacker toolkit', 'zero-day exploit']
  
  # Minimum content length: 300 characters
  # Stock listing filter: 'eod stock quote', 'historical prices', 'company symbol'
  ```

- **Deduplication**:
  - 80% similarity threshold
  - SequenceMatcher (title + first 200 chars)
  - Jaccard similarity (URL domains)

- **Competitor Detection**: 27+ competitors tracked
  ```python
  competitors = [
      'Experian', 'Equifax', 'LexisNexis', 'FICO', 'SAS',
      'Feedzai', 'DataVisor', 'Kount', 'Riskified', 'Forter', 'Sift', 'Signifyd',
      'Jumio', 'Onfido', 'Veriff', 'Trulioo', 'IDnow', 'Socure', 'Mitek',
      'BioCatch', 'Nuance', 'Shield', 'Sardine', 'Unit21', 'Alloy', 'Persona',
      'Microblink', 'Signicat', 'SITA', 'Indicio', 'iDAKTO', 'GET Group', 'Sumsub'
  ]
  ```

- **Fraud Domains**: 10 categories
  ```python
  fraud_keywords = {
      'identity_verification': ['identity verification', 'KYC', 'know your customer'],
      'biometrics': ['biometric', 'fingerprint', 'facial recognition', 'liveness detection'],
      'fraud_detection': ['fraud detection', 'fraud prevention', 'anti-fraud'],
      'account_takeover': ['account takeover', 'ATO', 'credential stuffing'],
      'synthetic_identity': ['synthetic identity', 'synthetic fraud'],
      'authentication': ['MFA', '2FA', 'passwordless', 'authentication'],
      'aml_compliance': ['AML', 'anti-money laundering', 'KYB'],
      'risk_analytics': ['risk analytics', 'risk assessment', 'risk scoring'],
      'device_intelligence': ['device fingerprinting', 'device intelligence'],
      'transaction_monitoring': ['transaction monitoring', 'payment fraud']
  }
  ```

- **Technology Keywords** (Fraud-Specific Only):
  ```python
  fraud_tech_keywords = [
      'biometric', 'facial recognition', 'fingerprint', 'liveness detection',
      'behavioral analytics', 'AI fraud', 'machine learning fraud',
      'blockchain identity', 'anomaly detection', 'neural network fraud'
  ]
  ```

#### 2. **PDF Report Generator** (`tools/improved_pdf_generator.py`)
**Purpose**: Generate professional PDF reports with charts and citations

**Key Features**:
- **Navy/Blue Corporate Theme** (matching original style from Oct 10, 2025)
- **4 Charts**:
  1. **Competitor Mentions Bar Chart** - Top 10 competitors (horizontal bars)
  2. **Competitive Threats Pie Chart** - 8 competitors with percentages
  3. **Technology Trends Bar Chart** - Fraud-specific tech keywords
  4. **Fraud Domain Distribution Bar Chart** - 10 fraud categories
- **10 Key Citations** - Article titles with clickable URLs
- **150-word Summaries** - AI-generated, ads/headers removed
- **Chart Specifications**:
  - 300 DPI high-resolution
  - Seaborn styling (`seaborn-v0_8-darkgrid`)
  - Values inside bars (white text) for large values, outside (black) for small
  - 15% x-axis margin to keep values inside frame

#### 3. **Output Files**

**Directory**: `reports/improved_ci_analysis_<timestamp>/`

**Generated Files**:
1. `final_ci_analysis.json` - Complete analysis data
   ```json
   {
       "metadata": {"time_period_days": 30, "total_articles": 92, "competitors_tracked": 27},
       "executive_summary": {"key_trends": [...], "emerging_threats": [...]},
       "competitor_activities": {"Persona": {"mention_count": 22, "sentiment": "positive"}},
       "fraud_domain_distribution": {"Account Takeover": 51, "AML": 30},
       "technology_trends": {"biometric": 23, "facial recognition": 9},
       "risk_assessment": {...}
   }
   ```

2. `deduplicated_articles.json` - All cleaned articles
   ```json
   [
       {
           "title": "...",
           "url": "...",
           "date": "2025-11-03",
           "content": "...",
           "summary": "...",
           "competitors_mentioned": ["Persona", "Onfido"],
           "fraud_domains": ["identity_verification", "biometrics"],
           "sentiment": "positive"
       }
   ]
   ```

3. `competitor_mentions_chart.png` - Bar chart (300 DPI)
4. `competitive_threats_pie_chart.png` - Pie chart (300 DPI)
5. `technology_trends_chart.png` - Bar chart (300 DPI)
6. `fraud_domain_distribution_chart.png` - Bar chart (300 DPI)
7. `Final_Competitive_Intelligence_Report_30Days.pdf` - Professional PDF report (500-600 KB)

---

## ⚙️ Configuration Details

### Key Configuration Variables

#### Scraping Configuration (`improved_ci_analyzer.py`)
```python
DAYS_TO_SCRAPE = 30              # Number of days to look back
SIMILARITY_THRESHOLD = 0.8       # Deduplication threshold (80%)
MIN_CONTENT_LENGTH = 300         # Minimum article length (characters)
CONCURRENT_SCRAPING = True       # Use async parallel scraping
TIMEOUT_SECONDS = 10             # HTTP request timeout
```

#### Report Configuration (`improved_pdf_generator.py`)
```python
SUMMARY_MAX_WORDS = 150          # Summary length
TOP_COMPETITORS_CHART = 10       # Number of competitors in bar chart
TOP_CITATIONS = 10               # Number of citations in PDF
CHART_DPI = 300                  # Chart resolution
COLOR_SCHEME = 'navy_blue'       # PDF theme
```

#### RSS Feed Sources (Customizable)
Located in `improved_ci_analyzer.py` line ~640:
```python
rss_feeds = [
    "https://www.biometricupdate.com/feed",
    "https://www.pymnts.com/feed/",
    "https://techcrunch.com/feed/",
    # ... add or remove feeds here
]
```

---

## 🚀 Execution Instructions

### Complete Workflow: 30-Day Competitive Intelligence Report

#### Step 1: Activate Virtual Environment
```powershell
# Windows PowerShell
cd competition_tracking_agent
.\venv\Scripts\Activate.ps1

# Linux/macOS
cd competition_tracking_agent
source venv/bin/activate
```

#### Step 2: Run Scraper and Analyzer
```bash
# This will:
# 1. Scrape 12 RSS feeds for last 30 days (80 seconds)
# 2. Filter junk articles (political, hacker, stock listings)
# 3. Deduplicate content (80% threshold)
# 4. Analyze competitors, fraud domains, technology trends
# 5. Save raw scrapes and analysis JSON

python tools/improved_ci_analyzer.py
```

**Expected Output**:
```
2025-11-04 14:30:15 - INFO - Starting competitive intelligence analysis
2025-11-04 14:30:15 - INFO - Scraping 12 RSS feeds for 30 days...
2025-11-04 14:31:35 - INFO - Scraped 131 articles in 80 seconds
2025-11-04 14:31:35 - INFO - Filtering junk articles...
2025-11-04 14:31:37 - INFO - Filtered out 39 junk articles
2025-11-04 14:31:37 - INFO - Deduplicating articles...
2025-11-04 14:31:38 - INFO - Removed 2 duplicate articles
2025-11-04 14:31:38 - INFO - Final count: 92 unique articles
2025-11-04 14:31:38 - INFO - Analyzing competitors and fraud domains...
2025-11-04 14:32:45 - INFO - Analysis complete
2025-11-04 14:32:45 - INFO - Saved to: reports/improved_ci_analysis_20251104_143245/
```

**Output Location**: `reports/improved_ci_analysis_<timestamp>/`

#### Step 3: Generate PDF Report
```bash
# This will:
# 1. Read analysis JSON from latest folder
# 2. Generate 4 charts (300 DPI PNG)
# 3. Create professional PDF report
# 4. Add 10 key citations

python tools/improved_pdf_generator.py
```

**Expected Output**:
```
2025-11-04 14:33:00 - INFO - Loading analysis data...
2025-11-04 14:33:01 - INFO - Found 92 articles to process
2025-11-04 14:33:02 - INFO - ✓ Competitor mentions chart created
2025-11-04 14:33:03 - INFO - ✓ Competitive threats pie chart created
2025-11-04 14:33:04 - INFO - ✓ Technology trends chart created
2025-11-04 14:33:05 - INFO - ✓ Fraud domain distribution chart created
2025-11-04 14:33:06 - INFO - Generating PDF report...
2025-11-04 14:33:12 - INFO - ✅ PDF report generated: Final_Competitive_Intelligence_Report_30Days.pdf
```

**Output Files**:
- `reports/improved_ci_analysis_<timestamp>/Final_Competitive_Intelligence_Report_30Days.pdf`
- 4 PNG charts in same folder

#### Step 4: Review Report
```bash
# Windows - Open PDF
start reports/improved_ci_analysis_*/Final_Competitive_Intelligence_Report_30Days.pdf

# macOS
open reports/improved_ci_analysis_*/Final_Competitive_Intelligence_Report_30Days.pdf

# Linux
xdg-open reports/improved_ci_analysis_*/Final_Competitive_Intelligence_Report_30Days.pdf
```

---

## 🎨 Key Features & Capabilities

### 1. **Intelligent Content Filtering**
- ✅ **Political Filter**: Removes NYPD, Trump administration, election articles
- ✅ **Hacker News Filter**: Removes Pegasus spyware, NSO Group, hacker toolkits
- ✅ **Stock Listing Filter**: Removes "EOD Stock Quote", "Historical Prices"
- ✅ **Ad Removal**: Strips "Get the Full Story", "Complete the form", subscription prompts
- ✅ **Length Filter**: Minimum 300 characters (removes snippet articles)

### 2. **AI-Powered Analysis**
- 🤖 **T5 Summarization**: 150-word executive summaries
- 🤖 **DistilBERT Sentiment**: Positive/Negative/Neutral classification
- 🤖 **BART Zero-Shot**: Fraud domain categorization (10 categories)

### 3. **Competitor Intelligence**
- 📊 **27+ Competitors Tracked**: Experian, Equifax, Jumio, Onfido, BioCatch, etc.
- 📊 **Mention Tracking**: Count of articles mentioning each competitor
- 📊 **Sentiment Analysis**: Positive/Negative tone for each mention
- 📊 **Product Launches**: Auto-detect "launched", "unveiled", "released"
- 📊 **Partnerships**: Auto-detect "partners", "collaboration", "acquired"

### 4. **Fraud Domain Analysis**
- 🛡️ **10 Categories**: Identity Verification, Biometrics, Fraud Detection, ATO, Synthetic Identity, Authentication, AML, Risk Analytics, Device Intelligence, Transaction Monitoring
- 🛡️ **Keyword Matching**: Each article tagged with relevant domains
- 🛡️ **Distribution Chart**: Visual representation of fraud landscape

### 5. **Technology Trends**
- 🔬 **Fraud-Specific Keywords**: Biometric (23), Facial Recognition (9), Fingerprint (7), Liveness Detection (2), Behavioral Analytics (2)
- 🔬 **Excludes Generic Terms**: No "API", "cloud", "data analytics" (unless fraud-specific)

### 6. **Professional Reporting**
- 📄 **Navy/Blue Theme**: Corporate style matching Oct 10, 2025 report
- 📄 **Clickable URLs**: All citations are hyperlinks
- 📄 **High-Res Charts**: 300 DPI for printing
- 📄 **Compact Summaries**: 150 words per article
- 📄 **10 Key Citations**: Most impactful articles

---

## 🐛 Troubleshooting

### Common Issues & Solutions

#### Issue 1: "Module not found" errors
```bash
# Solution: Ensure virtual environment is activated
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate     # Linux/macOS

# Reinstall dependencies
pip install -r requirements.txt
```

#### Issue 2: "Hugging Face model download fails"
```bash
# Solution: Check internet connection, increase timeout
# Manual download:
python -c "from transformers import T5Tokenizer, T5ForConditionalGeneration; T5Tokenizer.from_pretrained('t5-small'); T5ForConditionalGeneration.from_pretrained('t5-small')"
```

#### Issue 3: "No articles scraped" or "Empty RSS feeds"
```bash
# Solution: RSS feeds may be down or changed
# Check feed URLs manually:
curl https://www.biometricupdate.com/feed

# Update RSS feed list in improved_ci_analyzer.py (line ~640)
```

#### Issue 4: "Charts not rendering" or "PDF generation fails"
```bash
# Solution: Missing matplotlib backend
pip install --upgrade matplotlib pillow

# On Linux, may need:
sudo apt-get install python3-tk
```

#### Issue 5: "UnicodeDecodeError" when reading JSON
```bash
# Solution: Ensure UTF-8 encoding
# Already fixed in improved_ci_analyzer.py:
# json.dump(..., ensure_ascii=True)  # Forces ASCII encoding
```

#### Issue 6: "Execution policy error" (Windows PowerShell)
```powershell
# Solution: Allow script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then reactivate virtual environment
.\venv\Scripts\Activate.ps1
```

#### Issue 7: "Too many junk articles removed"
```python
# Solution: Adjust filtering thresholds in improved_ci_analyzer.py
# Line ~102 (junk_patterns): Comment out overly aggressive filters
# Line ~88 (MIN_CONTENT_LENGTH): Lower from 300 to 200 characters
```

#### Issue 8: "Charts show values outside frame"
```python
# Solution: Already fixed in improved_pdf_generator.py
# Ensure line ~250 has: plt.margins(x=0.15)  # 15% margin
```

---

## 📊 Sample Output Statistics

**Typical 30-Day Run** (Oct 4 - Nov 3, 2025):
- **Total Articles Scraped**: 131
- **Junk Articles Filtered**: 39 (30%)
  - Political: 12
  - Hacker/Spyware: 8
  - Stock Listings: 6
  - Short Content (<300 chars): 13
- **Duplicates Removed**: 2
- **Final Clean Articles**: 92

**Competitor Distribution**:
- Persona: 22 mentions
- SITA: 5 mentions
- Shield: 4 mentions
- SAS: 3 mentions
- Nuance: 3 mentions

**Fraud Domain Distribution**:
- Account Takeover: 51 articles
- AML/Compliance: 30 articles
- Biometrics: 26 articles

**Technology Trends**:
- Biometric: 23 mentions
- Facial Recognition: 9 mentions
- Fingerprint: 7 mentions
- Liveness Detection: 2 mentions

---

## 📞 Support & Maintenance

### Regular Maintenance Tasks

1. **Weekly**: Update RSS feed list if new sources emerge
2. **Monthly**: Review junk filter patterns (add new keywords if needed)
3. **Quarterly**: Update competitor list (add/remove companies)
4. **Annually**: Review fraud domain categories (adjust to industry trends)

### Performance Optimization

```python
# Increase scraping speed (improved_ci_analyzer.py line ~600)
CONCURRENT_REQUESTS = 20  # Default: 10 (increase for faster scraping)

# Reduce AI model loading time (use smaller models)
# improved_ci_analyzer.py line ~400
model_name = "t5-small"  # Options: t5-small (77MB), t5-base (242MB)
```

### Logs & Debugging

```bash
# Enable debug logging
# Add to improved_ci_analyzer.py line 23:
logging.basicConfig(level=logging.DEBUG)

# Check logs for scraping issues
grep "ERROR" logs/scraper.log

# Test individual RSS feed
python -c "import feedparser; print(feedparser.parse('https://www.pymnts.com/feed/').entries[0])"
```

---

## 📝 Quick Start Checklist

- [ ] Python 3.8.5+ installed (`python --version`)
- [ ] Virtual environment created (`python -m venv venv`)
- [ ] Virtual environment activated (`.\venv\Scripts\Activate.ps1`)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Imports tested (`python -c "import requests, feedparser, transformers"`)
- [ ] Scraper executed (`python tools/improved_ci_analyzer.py`)
- [ ] PDF generator executed (`python tools/improved_pdf_generator.py`)
- [ ] Report reviewed (`start reports/improved_ci_analysis_*/Final_*.pdf`)

---

## 🎯 Next Steps After Setup

1. **Customize Competitor List** - Edit `improved_ci_analyzer.py` line ~68
2. **Add RSS Feeds** - Update `rss_feeds` list in `improved_ci_analyzer.py` line ~640
3. **Adjust Filters** - Modify `junk_patterns` in `improved_ci_analyzer.py` line ~80
4. **Schedule Automation** - Use Windows Task Scheduler or cron for daily/weekly runs
5. **Integrate Dashboard** - Launch Streamlit UI: `streamlit run competition_agent/dashboard/app.py`

---

## 📚 Additional Resources

- **Hugging Face Docs**: https://huggingface.co/docs/transformers
- **ReportLab Docs**: https://www.reportlab.com/docs/reportlab-userguide.pdf
- **Feedparser Docs**: https://feedparser.readthedocs.io/
- **Matplotlib Docs**: https://matplotlib.org/stable/contents.html

---

**Last Updated**: November 4, 2025  
**Version**: 3.0 (Final Stable Release)  
**Contact**: biswaj45 (GitHub)

---

## 🔐 Security & Privacy Notes

⚠️ **No API Keys Required** - System uses public RSS feeds only  
⚠️ **Local Processing** - All AI models run locally (no cloud API calls)  
⚠️ **No Personal Data** - Scrapes only public news articles  
⚠️ **Ethical Use** - Respects robots.txt and rate limits RSS feeds

---

**End of Deployment Guide** ✅
