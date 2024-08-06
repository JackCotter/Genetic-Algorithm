from data_utils import read_dataset
from random import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

SCORE_THRESHOLD = 1
MUTATION_FACTOR = 0.3
ROUNDS = 100
PARENTS_PER_ROUND = 12
CHILDREN_PER_MATE = 4
CLASSIFIER_TRAINING_PERCENTAGE = 0.15
TESTING_DATA_PERCENTAGE = 0.15
NUMBER_OF_NEAREST_NEIGHBOURS = 7

def evaluation_function( review_obj, parent ):
  score_logreg = ( 1 if review_obj.get( 'logreg' ) else 0 ) * parent[ 'logreg_weight' ]
  score_knn = ( 1 if review_obj.get( 'knn' ) else 0 ) * parent[ 'knn_weight' ]
  score_svc = ( 1 if review_obj.get('svc') else 0 ) * parent[ 'svc_weight' ]
  score_nb = ( 1 if review_obj.get( 'nb' ) else 0 ) * parent[ 'nb_weight' ]

  if ( score_logreg + score_knn + score_svc + score_nb ) > SCORE_THRESHOLD:
    return True
  else:
    return False

def evaluate_fitness( parent, reviews ):
  num_reviews = 0
  correct = 0

  y_true = []
  y_pred = []
  for review in reviews:
    result = evaluation_function( review, parent )
    if result == review[ "fake" ]:
      correct += 1

    y_true.append( review[ "fake" ] )
    y_pred.append( result )
    num_reviews += 1

  cm = confusion_matrix( y_true, y_pred )
  return ( correct / num_reviews, cm )
    
def get_best_x_parents( parents, accuracies ):
  accuracies_to_parent = {}
  for cur_index, accuracy in enumerate( accuracies ):
    accuracies_to_parent[ accuracy ] = parents[ cur_index ]
  sorted_accuracies = accuracies
  sorted_accuracies.sort( reverse = True )
  best_parents = []
  for index in range( int( len( parents ) / 2 ) ):
    best_parents.append( accuracies_to_parent[ sorted_accuracies[ index ] ] )
  return best_parents

def mate_parents( parents ):
  def generate_mutation():
    # Returns MUTATION_FACTOR * (float in range -0.5, 0.5)
    return MUTATION_FACTOR * ( 0.5 - random() )

  index = 0
  new_parents = []
  while index + 1 < len( parents ):
    new_logreg_weight = ( parents[ index ][ 'logreg_weight' ] + parents[ index+1 ][ 'logreg_weight' ] ) / 2
    new_knn_weight = (parents[ index ][ 'knn_weight' ] + parents[ index + 1 ][ 'knn_weight' ] ) / 2
    new_svc_weight = ( parents[ index ][ 'svc_weight' ] + parents[ index + 1 ][ 'svc_weight' ] ) / 2
    new_nb_weight = ( parents[ index ][ 'nb_weight' ] + parents[ index ][ 'nb_weight' ] ) / 2
    temp_parents = [ {
      'id':index,
      'logreg_weight': new_logreg_weight + generate_mutation(),
      'knn_weight': new_knn_weight + generate_mutation(),
      'svc_weight': new_svc_weight + generate_mutation(),
      'nb_weight': new_nb_weight + generate_mutation()
    } for index in range( CHILDREN_PER_MATE ) ]
    new_parents += temp_parents
    index += 2
    
  return new_parents


def genetic_algorithm( reviews, testing_reviews, genetic_rounds ):
  learning_curve = []
  parents = [ { 
    'id' : index, 
    'logreg_weight' : random(), 
    'knn_weight' : random(), 
    'svc_weight' : random(), 
    'nb_weight' : random() 
  } for index in range( PARENTS_PER_ROUND ) ]
  for _ in range( genetic_rounds ):
    accuracies = []
    for parent in parents:
      accuracy, cm = evaluate_fitness( parent, reviews )
      accuracies.append( accuracy )
    
    best_parents = get_best_x_parents( parents, accuracies )
    print( len( best_parents ) )
    parents = mate_parents( best_parents )
    learning_curve.append( max( accuracies ) )

  max_value = max( accuracies )
  max_index = accuracies.index( max_value )
  parent_value = parents[ max_index ]
  accuracy, cm = evaluate_fitness( parent_value, testing_reviews )
  return ( 
    accuracy,
    cm,
    learning_curve
  )

def train_classifiers( reviews, vectorizer ):
  X_train = []
  y_train = []
  for review in reviews:
    X_train.append( review[ "review" ] )
    y_train.append( int( review[ "fake" ] ) )

  X_train = vectorizer.fit_transform( X_train )
  
  logreg = LogisticRegression()
  logreg.fit( X_train, y_train )

  knn = KNeighborsClassifier( n_neighbors = NUMBER_OF_NEAREST_NEIGHBOURS )
  knn.fit( X_train, y_train )

  svm = SVC( kernel = 'linear' )
  svm.fit( X_train, y_train )

  nb = MultinomialNB()
  nb.fit( X_train, y_train )

  return ( logreg, knn, svm, nb )

def run_classifiers( reviews, vectorizer, logreg, knn, svc, nb ):
  for review in reviews:
    transformed_review = vectorizer.transform([ review[ 'review' ] ] )
    review[ 'logreg' ] = logreg.predict( transformed_review )
    review[ 'knn' ] = knn.predict( transformed_review )
    review[ 'svc' ] = svc.predict( transformed_review )
    review[ 'nb' ] = nb.predict( transformed_review )
  return reviews

def main():
  reviews = read_dataset( "fake reviews dataset.csv", None )
  classifier_training_reviews = reviews[ 0 : int( len( reviews ) * CLASSIFIER_TRAINING_PERCENTAGE ) ]
  genetic_algorithm_training_reviews = reviews[ int( len( reviews ) * CLASSIFIER_TRAINING_PERCENTAGE ) ]
  
  vectorizer = CountVectorizer()
  logreg, knn, svc, nb = train_classifiers( classifier_training_reviews, vectorizer )
  genetic_algorithm_training_reviews = run_classifiers( classifier_training_reviews, vectorizer, logreg, knn, svc, nb )
  testing_reviews = genetic_algorithm_training_reviews[ int( -1 * len( reviews ) * TESTING_DATA_PERCENTAGE ): ]
  genetic_algorithm_training_reviews = genetic_algorithm_training_reviews[ :int( 1 * len( reviews ) * TESTING_DATA_PERCENTAGE ) ]
  accuracy, cm, learning_curve = genetic_algorithm( genetic_algorithm_training_reviews, testing_reviews, ROUNDS )
  print( f'In the dataset of { len( reviews ) } reviews' )
  print( f'where { ( 1 - TESTING_DATA_PERCENTAGE - CLASSIFIER_TRAINING_PERCENTAGE ) * 100 }% were used for training' )
  print( f'and { TESTING_DATA_PERCENTAGE * 100 }% were used for testing' )
  print( f'{ round( accuracy * 100, 4 ) }% of the tests were predicted accurately')

  print( 'Confusion Matrix' )
  print( cm )

  print( 'Learning Curve' )
  counter = 1;
  while( counter < len( learning_curve ) ):
    print( learning_curve[ counter ] - learning_curve[ counter - 1 ] )
    counter += 1;

if __name__=='__main__':
  main()


