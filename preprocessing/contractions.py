"""
A small contractions dictionary.

NLTK doesn't ship one out of the box, so we keep it here. Add as needed -
Twitter has a long tail of casual contractions ('ya, dunno, gonna...).
"""

CONTRACTIONS = {
    "ain't": "is not", "aren't": "are not", "can't": "cannot",
    "could've": "could have", "couldn't": "could not",
    "didn't": "did not", "doesn't": "does not", "don't": "do not",
    "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
    "he'd": "he would", "he'll": "he will", "he's": "he is",
    "how'd": "how did", "how's": "how is",
    "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have",
    "isn't": "is not", "it'd": "it would", "it'll": "it will", "it's": "it is",
    "let's": "let us",
    "mustn't": "must not", "might've": "might have",
    "shan't": "shall not", "she'd": "she would", "she'll": "she will",
    "she's": "she is", "should've": "should have", "shouldn't": "should not",
    "that's": "that is", "there's": "there is", "they'd": "they would",
    "they'll": "they will", "they're": "they are", "they've": "they have",
    "wasn't": "was not", "we'd": "we would", "we'll": "we will",
    "we're": "we are", "we've": "we have", "weren't": "were not",
    "what'll": "what will", "what're": "what are", "what's": "what is",
    "what've": "what have", "where's": "where is",
    "who'd": "who would", "who'll": "who will", "who's": "who is",
    "who've": "who have",
    "won't": "will not", "would've": "would have", "wouldn't": "would not",
    "y'all": "you all", "you'd": "you would", "you'll": "you will",
    "you're": "you are", "you've": "you have",
    # casual / Twitter-flavoured
    "gonna": "going to", "wanna": "want to", "gotta": "got to",
    "dunno": "do not know", "lemme": "let me", "gimme": "give me",
}


def expand_contractions(text: str) -> str:
    """Lowercases-then-expands every contraction we know about. Anything we
    don't have a mapping for is left alone, which is the safe default."""
    out = []
    for token in text.split():
        key = token.lower()
        out.append(CONTRACTIONS.get(key, token))
    return " ".join(out)
