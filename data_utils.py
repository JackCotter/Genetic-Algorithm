import csv

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

def read_dataset(dataset_path):
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
          if buffer.get('type') == 'OR':
            real_reviews.append({
              'rating': buffer.get('rating'),
              'review': buffer.get('review')
            })
          elif buffer.get('type') == 'CG':
            fake_reviews.append({
              'rating': buffer.get('rating'),
              'review': buffer.get('review')
            })
          buffer = {
            'rating': row[1],
            'review': row[3],
            'type': row[2]
          }
        else:
          buffer['review'] += row[0]
      except:
        rows_read_incorrectly += 1
    if rows_read_incorrectly > 0:
      print("read " + str(rows_read_incorrectly) + " rows incorrectly")
    return (real_reviews, fake_reviews)

