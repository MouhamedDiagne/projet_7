"""Portable implementation of the notebook's configured tweet preprocessing."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

import emoji
from contractions import fix

_EMAIL = re.compile(r'(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b')
_URL = re.compile(r'(?i)\b(?:https?://|www\.)\S+')
_MENTION = re.compile(r'(?<!\w)@\w+')
_HASHTAG = re.compile(r'(?<!\w)#(\w+)')
_NUMBER = re.compile(r'\b\d+(?:[.,]\d+)?\b')
_PUNCTUATION = re.compile(r'[^\w\s]')
_EMOTICONS = (
    (re.compile(r"(?i)(?:[:=;8]['-]?[)d]+|[)d]+['-]?[:=;8])"), ' smile_token '),
    (re.compile(r"(?i)[:=;8]['-]?p+"), ' laugh_token '),
    (re.compile(r"(?i)(?:[:=;8]['-]?\(+|\)+['-]?[:=;8])"), ' sadface_token '),
    (re.compile(r"(?i)[:=;8]['-]?(?:[/|l*])"), ' neutralface_token '),
)


@dataclass(frozen=True)
class PreprocessingConfig:
    """Structured equivalent of the benchmark preprocessing artifact."""

    profile: str = 'whitespace_v1'
    normalize_whitespace: bool = True
    decode_html_entities: bool = False
    expand_contractions: bool = False
    replace_urls: bool = False
    replace_emails: bool = False
    replace_mentions: bool = False
    expand_hashtags: bool = False
    demojize_emojis: bool = False
    replace_ascii_emoticons: bool = False
    reduce_elongated_tokens: bool = False
    replace_numbers: bool = False
    encode_sentiment_punctuation: bool = False
    remove_digits: bool = False
    remove_punctuation: bool = False

    @classmethod
    def from_mlflow_payload(cls, payload: dict) -> 'PreprocessingConfig':
        """Read the flattened preprocessing JSON artifact emitted by the notebook."""
        fields = cls.__dataclass_fields__
        values = {}
        for name in fields:
            value = payload.get(f'preprocessing_{name}', payload.get(name, fields[name].default))
            if isinstance(fields[name].default, bool) and isinstance(value, str):
                value = value.lower() == 'true'
            values[name] = value
        return cls(**values)


def preprocess_tweet(tweet: str, config: PreprocessingConfig) -> str:
    """Apply the same ordered text transformations as the benchmark notebook."""
    text = str(tweet)
    if config.decode_html_entities:
        text = html.unescape(text)
    if config.replace_emails:
        text = _EMAIL.sub(' email_token ', text)
    if config.replace_urls:
        text = _URL.sub(' url_token ', text)
    if config.replace_mentions:
        text = _MENTION.sub(' user_token ', text)
    if config.expand_hashtags:
        text = _HASHTAG.sub(r' hashtag_token \1 ', text)
    if config.expand_contractions:
        text = fix(text)
    if config.demojize_emojis:
        text = emoji.demojize(text, delimiters=(' emoji_', ' '))
    if config.replace_ascii_emoticons:
        for pattern, replacement in _EMOTICONS:
            text = pattern.sub(replacement, text)
        text = text.replace('<3', ' heart_token ')
    if config.reduce_elongated_tokens:
        text = re.sub(r'([A-Za-z])\1{2,}', r'\1\1', text)
    if config.replace_numbers:
        text = _NUMBER.sub(' number_token ', text)
    elif config.remove_digits:
        text = re.sub(r'\d+', ' ', text)
    if config.encode_sentiment_punctuation:
        text = re.sub(r'!{2,}', ' repeated_exclamation_token ', text)
        text = re.sub(r'\?{2,}', ' repeated_question_token ', text)
        text = text.replace('!', ' exclamation_token ').replace('?', ' question_token ')
    elif config.remove_punctuation:
        text = _PUNCTUATION.sub(' ', text)
    return re.sub(r'\s+', ' ', text).strip() if config.normalize_whitespace else text.strip()
