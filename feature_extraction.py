import re
from urllib.parse import urlparse
import tldextract
import email
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# URL Feature Extraction
def extract_url_features(url):
    features = {}
    
    # URL length
    features['url_length'] = len(url)
    
    # Parse URL
    parsed_url = urlparse(url)
    extract_result = tldextract.extract(url)
    
    # Domain features
    features['domain_length'] = len(extract_result.domain)
    features['subdomain_length'] = len(extract_result.subdomain)
    features['tld'] = extract_result.suffix
    
    # Count special characters
    features['num_dots'] = url.count('.')
    features['num_dashes'] = url.count('-')
    features['num_underscores'] = url.count('_')
    features['num_at_signs'] = url.count('@')
    features['num_ampersands'] = url.count('&')
    features['num_question_marks'] = url.count('?')
    features['num_equal_signs'] = url.count('=')
    
    # Check for suspicious patterns
    features['has_ip_address'] = 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0
    features['has_https'] = 1 if url.startswith('https://') else 0
    
    return features

# Email Feature Extraction
nltk.download('punkt')
nltk.download('stopwords')

def extract_email_features(email_content):
    features = {}
    
    # Parse email
    msg = email.message_from_string(email_content)
    
    # Header features
    features['has_reply_to'] = 1 if msg.get('Reply-To') else 0
    features['sender_domain'] = msg.get('From', '').split('@')[-1] if '@' in msg.get('From', '') else ''
    
    # Subject features
    subject = msg.get('Subject', '')
    features['subject_length'] = len(subject)
    features['subject_has_urgent'] = 1 if re.search(r'urgent|important|alert|attention', subject.lower()) else 0
    
    # Body features
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain" or content_type == "text/html":
                body += part.get_payload(decode=True).decode('utf-8', errors='ignore')
    else:
        body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
    
    # Clean HTML
    if '<html' in body.lower():
        soup = BeautifulSoup(body, 'html.parser')
        body = soup.get_text()
    
    # Text features
    tokens = word_tokenize(body.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if w not in stop_words]
    
    features['body_length'] = len(body)
    features['num_links'] = body.lower().count('href=')
    features['has_password_mention'] = 1 if re.search(r'password|credential|login|sign in', body.lower()) else 0
    
    return features