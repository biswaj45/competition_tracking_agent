# Known subscription/paywall domains
PAYWALL_DOMAINS = {
    'wsj.com',  # Wall Street Journal
    'ft.com',   # Financial Times
    'bloomberg.com',
    'reuters.com',
    'economist.com',
    'barrons.com',
    'seekingalpha.com',
    'law360.com',
    'americanbanker.com'
}

# Sites requiring special handling
CUSTOM_HANDLERS = {
    'techcrunch.com': 'article',
    'venturebeat.com': 'article',
    'zdnet.com': 'article.article',
    'businesswire.com': 'bw-article__content',
    'prnewswire.com': 'release-text',
    'finextra.com': 'article-content'
}