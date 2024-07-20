from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle

model = MultinomialNB()
vectorizer = None

def train( data ):
    global model, vectorizer

    X = data[ 'text_' ]
    y = data[ 'label' ]

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 42 )

    vectorizer = TfidfVectorizer( max_features = 5000 )
    X_train_vec = vectorizer.fit_transform( X_train )

    model.fit( X_train_vec, y_train )

    with open('model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    with open('vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    return

def predict( review ):
    global model, vectorizer
        
    if model is None or vectorizer is None:
        with open( 'model.pkl', 'rb' ) as model_file:
            model = pickle.load(model_file)
        with open( 'vectorizer.pkl', 'rb' ) as vectorizer_file:
            vectorizer = pickle.load( vectorizer_file )
    
    review_vec = vectorizer.transform( [review] )
    prediction = model.predict(review_vec)
    
    print( prediction )
    return prediction[0]