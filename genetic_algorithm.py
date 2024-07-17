from data_utils import read_dataset
from random import random

LENGTH_OF_DATASET = 100
SCORE_THRESHOLD = 10
MUTATION_FACTOR = 0.2
ROUNDS = 5
PARENTS_PER_ROUND = 10


def evaluation_function(review_obj, parent):
  if (parent['missp_words_weight'] * review_obj.get('num_missp_words') + parent['num_words_weight'] * review_obj.get('num_words'))> SCORE_THRESHOLD:
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
  new_parents = [{'id':index,
                  'missp_words_weight': new_missp_words_weight + generate_mutation(),
                  'num_words_weight': new_num_words_weight + generate_mutation()} for index in range(PARENTS_PER_ROUND)]
  return new_parents


def genetic_algorithm(real_reviews, fake_reviews, genetic_rounds):
  parents = [{'id':index, 'missp_words_weight': random(), 'num_words_weight': random()} for index in range(PARENTS_PER_ROUND)]
  for _ in range(genetic_rounds):
    accuracies = []
    for parent in parents:
      accuracy = evaluate_fitness(parent, real_reviews, fake_reviews)
      accuracies.append(accuracy)
    
    (parent1, parent2) = get_best_two_parents(parents, accuracies)
    
    parents = mate_parents(parent1, parent2)

  return accuracies

def main():
  real_reviews, fake_reviews = read_dataset("fake reviews dataset.csv", LENGTH_OF_DATASET)
  accuracies = genetic_algorithm(real_reviews, fake_reviews, ROUNDS)
  print(accuracies)

if __name__=='__main__':
  main()


