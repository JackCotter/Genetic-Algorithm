from data_utils import read_dataset
from random import random
from spellchecker import SpellChecker

MISSPELLED_WORDS_THRESHOLD = 10

def number_missp_words(reviewText):
  spell = SpellChecker()
  words = reviewText.split()
  misspelled = spell.unknown(words)
  return len(misspelled)

def evaluation_function(review_obj, parent):
  if parent['missp_words_weight'] * number_missp_words(review_obj.get('review')) > MISSPELLED_WORDS_THRESHOLD:
    return True
  else:
    return False

def evaluate_fitness(parent, real_reviews, fake_reviews):
  num_reviews = 0
  correct = 0
  for review in real_reviews:
    result = evaluation_function(review, parent)
    if result == False:
      correct += 1
    num_reviews += 1
  for review in fake_reviews:
    result = evaluation_function(review, parent)
    if result == True:
      correct += 1
    num_reviews += 1
  return correct / num_reviews
    

def genetic_algorithm(real_reviews, fake_reviews):
  parents = [{'id':index, 'missp_words_weight': random()} for index in range(10)]
  accuracies = []
  
  for parent in parents:
    accuracy = evaluate_fitness(parent, real_reviews, fake_reviews)
    accuracies.append(accuracy)

  return accuracies

def main():
  real_reviews, fake_reviews = read_dataset("fake reviews dataset.csv")
  accuracies = genetic_algorithm(real_reviews, fake_reviews)
  print(accuracies)

if __name__=='__main__':
  main()


