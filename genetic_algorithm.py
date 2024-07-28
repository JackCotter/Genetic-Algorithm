from data_utils import read_dataset
from random import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

LENGTH_OF_DATASET = 1000
SCORE_THRESHOLD = 1
MUTATION_FACTOR = 0.2
ROUNDS = 5
PARENTS_PER_ROUND = 100
CLASSIFIER_TRAINING_PERCENTAGE = 0.15
TESTING_DATA_PERCENTAGE = 0.15
NUMBER_OF_NEAREST_NEIGHBOURS = 3

def evaluation_function(review_obj, parent):
  score_logreg = (1 if review_obj.get('logreg') else 0) * parent['logreg_weight']
  score_knn = (1 if review_obj.get('knn') else 0) * parent['knn_weight']
  score_svc = (1 if review_obj.get('svc') else 0) * parent['svc_weight']

  if (score_logreg + score_knn + score_svc) > SCORE_THRESHOLD:
    return True
  else:
    return False

def evaluate_fitness(parent, reviews):
  num_reviews = 0
  correct = 0
  fake_counter = 0
  real_counter = 0
  for review in reviews:
    result = evaluation_function(review, parent)
    if result == review["fake"]:
      correct += 1
    if result:
      fake_counter += 1
    else:
      real_counter += 1
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

  new_logreg_weight = (parent1['logreg_weight'] + parent2['logreg_weight']) / 2
  new_knn_weight = (parent1['knn_weight'] + parent2['knn_weight']) / 2
  new_svc_weight = (parent1['svc_weight'] + parent2['svc_weight']) / 2
  new_parents = [{'id':index,
                  'logreg_weight': new_logreg_weight + generate_mutation(),
                  'knn_weight': new_knn_weight + generate_mutation(),
                  'svc_weight': new_svc_weight + generate_mutation(),
                  } for index in range(PARENTS_PER_ROUND)]
  return new_parents


def genetic_algorithm(reviews, testing_reviews, genetic_rounds):
  parents = [{'id':index, 'logreg_weight': random(), 'knn_weight': random(), 'svc_weight': random()} for index in range(PARENTS_PER_ROUND)]
  for _ in range(genetic_rounds):
    accuracies = []
    for parent in parents:
      accuracy = evaluate_fitness(parent, reviews)
      accuracies.append(accuracy)
    
    (parent1, parent2) = get_best_two_parents(parents, accuracies)
    
    parents = mate_parents(parent1, parent2)

  max_value = max(accuracies)
  max_index = accuracies.index(max_value)
  parent_value = parents[max_index]
  accuracy = evaluate_fitness(parent_value, testing_reviews)
  return (accuracy, parent_value['logreg_weight'], parent_value['knn_weight'], parent_value['svc_weight'])

def train_classifiers(reviews, vectorizer):
  X_train = []
  y_train = []
  for review in reviews:
    X_train.append(review["review"])
    y_train.append(int(review["fake"]))

  X_train = vectorizer.fit_transform(X_train)
  
  logreg = LogisticRegression()
  logreg.fit(X_train, y_train)

  knn = KNeighborsClassifier(n_neighbors=NUMBER_OF_NEAREST_NEIGHBOURS)
  knn.fit(X_train, y_train)

  svm = SVC(kernel='linear')
  svm.fit(X_train, y_train)

  return (logreg, knn, svm)

def run_classifiers(reviews, vectorizer, logreg, knn, svc):
  for review in reviews:
    transformed_review = vectorizer.transform([review['review']])
    review['logreg'] = logreg.predict(transformed_review)
    review['knn'] = knn.predict(transformed_review)
    review['svc'] = svc.predict(transformed_review)
  return reviews

def main():
  reviews = read_dataset("fake reviews dataset.csv", None)
  classifier_training_reviews = reviews[0:int(len(reviews) * CLASSIFIER_TRAINING_PERCENTAGE)]
  genetic_algorithm_training_reviews = reviews[int(len(reviews) * CLASSIFIER_TRAINING_PERCENTAGE)]
  
  vectorizer = CountVectorizer()
  logreg, knn, svm = train_classifiers(classifier_training_reviews, vectorizer)
  genetic_algorithm_training_reviews = run_classifiers(classifier_training_reviews, vectorizer, logreg, knn, svm)
  testing_reviews = genetic_algorithm_training_reviews[int(-1 * len(reviews) * TESTING_DATA_PERCENTAGE):]
  genetic_algorithm_training_reviews = genetic_algorithm_training_reviews[:int(1 * len(reviews) * TESTING_DATA_PERCENTAGE)]
  accuracy, logreg_weight, knn_weight, svc_weight = genetic_algorithm(genetic_algorithm_training_reviews, testing_reviews, ROUNDS)
  print(accuracy)
  print(logreg_weight)
  print(knn_weight)
  print(svc_weight)

if __name__=='__main__':
  main()


