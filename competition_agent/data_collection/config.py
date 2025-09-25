"""
Configuration for data collection sources
"""
# Competitor categories and their members
COMPETITORS = {
    "established": [
        "Experian Hunter",
        "LexisNexis Risk Solutions",
        "FICO Falcon",
        "Equifax",
        "ACI Worldwide"
    ],
    "mid_sized": [
        "Feedzai",
        "Featurespace",
        "BioCatch",
        "Emailage",
        "Kount"
    ],
    "startups": [
        "Sardine",
        "SentiLink",
        "SEON",
        "Ravelin",
        "Alloy"
    ]
}

# Company websites and their RSS/news feeds
COMPANY_FEEDS = {
    "Experian": {
        "news_url": "https://www.experian.com/blogs/news/category/fraud-identity/feed/",
        "blog_url": "https://www.experian.com/blogs/insights/category/fraud-and-identity/feed/"
    },
    "LexisNexis": {
        "news_url": "https://risk.lexisnexis.com/insights/press-release",
        "blog_url": "https://risk.lexisnexis.com/insights/blog"
    },
    # Add more company feeds here
}

# Tech media sources
TECH_MEDIA = [
    {
        "name": "Finextra",
        "url": "https://www.finextra.com/rss/channel.aspx?channel=fraud"
    },
    {
        "name": "PYMNTS",
        "url": "https://www.pymnts.com/category/security-and-risk/feed/"
    }
]

# Keywords for filtering relevant content
KEYWORDS = [
    "fraud detection",
    "fraud prevention",
    "fraud analytics",
    "anti-fraud",
    "identity verification",
    "KYC",
    "AML",
    "transaction monitoring",
    "behavioral biometrics",
    "risk assessment"
]