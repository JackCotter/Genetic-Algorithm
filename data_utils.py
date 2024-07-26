import csv

from spellchecker import SpellChecker

titles = [
  'Home_And_Kitchen',
  'Sports_and_Outdoors',
  'Electronics',
  'Movies_and_TV',
  'Tools_and_Home_Improvement',
  'Pet_Supplies',
  'Kindle_Store',
  'Books',
  'Toys_and_Games',
  'Clothing_Shoes_and_Jewelry'
]

def is_new_entry(entry):
  for title in titles:
    if title in entry:
      return True
  return False

def number_missp_words(reviewText):
  return 0
  # TODO As of right now, no missp words are found in the dataset, so this always returns 0. Either need to find a new dataset, or use a different metric. return 0 for now to speed up.
  spell = SpellChecker()
  words = reviewText.split()
  misspelled = spell.unknown(words)
  return len(misspelled) if misspelled is not None else 0

def number_words_total(reviewText):
  words = reviewText.split()
  return len(words)

def number_matching_keywords(reviewText, keyword_dict):
  words = reviewText.split()
  matching_keywords = 0
  for word in words:
    if keyword_dict.get(word) is not None:
      matching_keywords += 1
  return matching_keywords

def read_dataset(dataset_path, reviews_used=None, keywords=[]):
  with open(dataset_path, mode='r', newline='') as csvfile:
    keyword_dict = {keyword: True for keyword in keywords} # init keyword dict for faster lookup times.
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    read_reviews = []
    rows_read_incorrectly = 0
    buffer = {
      'rating': -1,
      'review': '',
      'type': ''
    }
    for row in reader:
      if not row:
        continue
      try:
        if is_new_entry(row[0]):
          final_review_obj = {
            'rating': buffer.get('rating'),
            'review': buffer.get('review'),
            'num_missp_words': number_missp_words(buffer.get('review')),
            'num_words': number_words_total(buffer.get('review')),
            'num_keywords': number_matching_keywords(buffer.get('review'), keyword_dict)
          }
          if buffer.get('type') == 'OR':
            final_review_obj["fake"] = False
          elif buffer.get('type') == 'CG':
            final_review_obj["fake"] = True
          else:
            final_review_obj = None
          if final_review_obj:
            read_reviews.append(final_review_obj)
          if reviews_used and len(read_reviews) > reviews_used:
            break;
          buffer = {
            'rating': row[1],
            'review': row[3],
            'type': row[2]
          }
        else:
          additional_text = ''
          for reviewText in row:
            additional_text += reviewText
          buffer['review'] += additional_text
      except:
        rows_read_incorrectly += 1
    if rows_read_incorrectly > 0:
      print("read " + str(rows_read_incorrectly) + " rows incorrectly")
    return read_reviews

