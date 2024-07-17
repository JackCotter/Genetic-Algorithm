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
  spell = SpellChecker()
  words = reviewText.split()
  misspelled = spell.unknown(words)
  return len(misspelled) if misspelled is not None else 0

def number_words_total(reviewText):
  words = reviewText.split()
  return len(words)

def read_dataset(dataset_path, reviews_used=None):
  with open(dataset_path, mode='r', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    real_reviews = []
    fake_reviews = []
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
            'num_words': number_words_total(buffer.get('review'))
          }
          if buffer.get('type') == 'OR':
            real_reviews.append(final_review_obj)
          elif buffer.get('type') == 'CG':
            fake_reviews.append(final_review_obj)
          print(buffer.get('review'))
          print(buffer.get('num_missp_words'))
          if reviews_used and len(real_reviews) + len(fake_reviews) > reviews_used:
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
    return (real_reviews, fake_reviews)

