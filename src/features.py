# src/features.py
import re
from urllib.parse import urlparse
import tldextract


IP_PATTERN = re.compile(r"^(?:http[s]?://)?(?:\d{1,3}\.){3}\d{1,3}")


def has_ip_address(url: str) -> int:
return int(bool(IP_PATTERN.search(url)))


def count_digits(s: str) -> int:
return sum(c.isdigit() for c in s)


def count_chars(s: str, chars: str) -> int:
return sum(s.count(c) for c in chars)


def extract_features(url: str) -> dict:
"""
Extracts a set of simple, fast URL-based features. Returns a dict.
"""
if not isinstance(url, str):
url = str(url)
p = urlparse(url)
ext = tldextract.extract(url)


host = p.netloc or ext.registered_domain or ''
path = p.path or ''
query = p.query or ''


features = {}
features['url_len'] = len(url)
features['host_len'] = len(host)
features['path_len'] = len(path)
features['count_dot'] = host.count('.')
features['count_subdomain_parts'] = len([p for p in host.split('.') if p])
features['has_https'] = int(p.scheme == 'https')
features['has_ip'] = has_ip_address(host)
features['count_hyphen'] = count_chars(url, '-')
features['count_at'] = count_chars(url, '@')
features['count_question'] = count_chars(url, '?')
features['count_equal'] = count_chars(url, '=')
features['count_percent'] = count_chars(url, '%')
features['count_underscore'] = count_chars(url, '_')
features['count_slash'] = count_chars(url, '/')
features['count_digits'] = count_digits(url)


# suspicious tokens
suspicious_tokens = ['login', 'signin', 'bank', 'secure', 'update', 'confirm', 'account']
for tok in suspicious_tokens:
features[f'token_{tok}'] = int(tok in url.lower())


# has port
features['has_port'] = int(':' in p.netloc and p.netloc.split(':')[-1].isdigit())


# tld length
features['tld_len'] = len(ext.suffix or '')


return features


# helper to convert series of urls -> DataFrame can be done in train.py
