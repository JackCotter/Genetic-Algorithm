from data_utils import read_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score

CLASSIFIER_TRAINING_PERCENTAGE = 0.15
TESTING_DATA_PERCENTAGE = 0.15
NUMBER_OF_NEAREST_NEIGHBOURS = 7
DATASET = 'fake reviews dataset.csv'


def training( reviews, vectorizer ):
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


def predict( reviews, vectorizer, logreg, knn, svc, nb ):
    for review in reviews:
        transformed_review = vectorizer.transform([ review[ 'review' ] ] )
        review[ 'logreg' ] = logreg.predict( transformed_review )[ 0 ]
        review[ 'knn' ] = knn.predict( transformed_review )[ 0 ]
        review[ 'svc' ] = svc.predict( transformed_review )[ 0 ]
        review[ 'nb' ] = nb.predict( transformed_review )[ 0 ]
    return reviews


def analyze( test_results ):
    y_true = [ int( review[ 'fake' ]) for review in test_results]
    classifiers = [ 'logreg', 'knn', 'svc', 'nb' ]
    results = {}

    for clf in classifiers:
        y_pred = [ int( review[ clf ] ) for review in test_results ]
        accuracy = accuracy_score( y_true, y_pred )
        conf_matrix = confusion_matrix(y_true, y_pred)
        results[ f'{ clf }_accuracy' ] = round( accuracy * 100, 4 )
        results[ f'{ clf }_confusion_matrix' ] = conf_matrix

    return results

def print_results( results ):
    classifiers = [ 'logreg', 'knn', 'svc', 'nb' ]

    for clf in classifiers:
        title = clf.upper()
        print( f'\n--- { title } ---' )
        print( f'Accuracy: { results[ f'{ clf }_accuracy' ] }%' )
        print( 'Confusion Matrix:' )
        print( results[ f'{ clf }_confusion_matrix' ] )
    return


def main():
    reviews = read_dataset( DATASET )
    vectorizer = CountVectorizer()
    training_reviews = reviews[ 0 : int( len( reviews ) * CLASSIFIER_TRAINING_PERCENTAGE  ) ]

    logreg, knn, svc, nb = training( training_reviews, vectorizer )

    test_reviews = reviews[ int( -1 * len( reviews ) * TESTING_DATA_PERCENTAGE ): ]
    test_results = predict( test_reviews, vectorizer, logreg, knn, svc, nb )

    results = analyze( test_results )
    print_results( results )

    return 0

if __name__=='__main__':
  main()