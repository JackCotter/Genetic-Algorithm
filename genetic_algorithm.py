from data_utils import read_dataset
from random import random

LENGTH_OF_DATASET = 1000
SCORE_THRESHOLD = 10
MUTATION_FACTOR = 0.2
ROUNDS = 5
PARENTS_PER_ROUND = 100
KEYWORDS = ['simple', 'appendix', 'nice', 'optics', 'last', 'bought', 'balls', 'cleats', 'update']

def evaluation_function(review_obj, parent):
  if (parent['missp_words_weight'] * review_obj.get('num_missp_words') + parent['num_words_weight'] * review_obj.get('num_words') 
      + parent['keywords_weight'] * review_obj.get('num_keywords')) > SCORE_THRESHOLD:
    return True
  else:
    return False

def evaluate_fitness(parent, reviews):
  num_reviews = 0
  correct = 0
  for review in reviews:
    result = evaluation_function(review, parent)
    if result == review["fake"]:
      correct += 1
    num_reviews += 1
  return correct / num_reviews
    
def get_best_two_parents(parents, accuracies):
  index_best_1 = 0
  index_best_2 = 0
  max_accuracy = 0
  for cur_index, accuracy in enumerate(accuracies):
    if accuracy > max_accuracy:
      index_best_2 = index_best_1
      index_best_1 = cur_index
  return (parents[index_best_1], parents[index_best_2])

def mate_parents(parent1, parent2):
  def generate_mutation():
    # Returns MUTATION_FACTOR * (float in range -0.5, 0.5)
    return MUTATION_FACTOR * (0.5 - random())

  new_missp_words_weight = (parent1['missp_words_weight'] + parent2['missp_words_weight']) / 2
  new_num_words_weight = (parent1['num_words_weight'] + parent2['num_words_weight']) / 2
  new_keywords_weight = (parent1['keywords_weight'] + parent2['keywords_weight']) / 2
  new_keywords_weight = (parent1['classifier_weight'] + parent2['classifier_weight']) / 2
  new_parents = [{'id':index,
                  'missp_words_weight': new_missp_words_weight + generate_mutation(),
                  'num_words_weight': new_num_words_weight + generate_mutation(),
                  'keywords_weight': new_keywords_weight + generate_mutation(),
                  'classifier_weight': new_keywords_weight + generate_mutation()
                  } for index in range(PARENTS_PER_ROUND)]
  return new_parents


def genetic_algorithm(reviews, genetic_rounds):
  parents = [{'id':index, 'missp_words_weight': random(), 'num_words_weight': random(), 'keywords_weight': random(), 'classifier_weight': random()} for index in range(PARENTS_PER_ROUND)]
  for _ in range(genetic_rounds):
    accuracies = []
    for parent in parents:
      accuracy = evaluate_fitness(parent, reviews)
      accuracies.append(accuracy)
    
    (parent1, parent2) = get_best_two_parents(parents, accuracies)
    
    parents = mate_parents(parent1, parent2)

  return accuracies

def main():
  reviews = read_dataset("fake reviews dataset.csv", None, KEYWORDS)
  accuracies = genetic_algorithm(reviews, ROUNDS)
  print(accuracies)

if __name__=='__main__':
  main()


