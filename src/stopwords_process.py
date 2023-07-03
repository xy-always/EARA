from nltk.tokenize import word_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords

def remove_stopwords(sent, stop_word_set='stanford_core'):
  stop_word_lists = ['ranks', 'nltk' , 'stanford_core']
  RANKS_STOPWORDS = "/disk4/xy/PROJECT/2019n2c2/processed_data/ranks_stopwords.txt"
  STANFORD_STOPWORDS = "/disk4/xy/PROJECT/2019n2c2/processed_data/stanford_core_stopwords.txt"
  nltk_stopwords = stopwords.words('english')
  ranks_stopwords = []
  stanford_core_stopwords = []

  with open(RANKS_STOPWORDS, 'r') as words:
      for i in words:
          ranks_stopwords.append(i.strip())

  with open(STANFORD_STOPWORDS, 'r') as words:
      for i in words:
          stanford_core_stopwords.append(i.strip())
  """Remove stopwords and punctuation"""
    # Note: BIOSSES original paper used stop words from https://www.ranks.nl/stopwords
    # and following punctuation: (.,!;/-?: colon, mark,)
  if stop_word_set == 'ranks':
      stop_words = set(ranks_stopwords)
  elif stop_word_set == 'stanford_core':
      stop_words = set(stanford_core_stopwords) 
  elif stop_word_set == 'nltk':
      stop_words = set(stopwords.words('english')) 
  punctuation_tokens = set('.,-!;/?:')
  word_tokens = wordpunct_tokenize(sent)
  filtered_sentence = [w for w in word_tokens if not (w in stop_words or w in punctuation_tokens)]
  filtered_sentence = ' '.join(filtered_sentence)
  #filtered_sentence = stemmer.stem(filtered_sentence)
  return filtered_sentence