import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import joblib


class SentimentClassifier:
    def __init__(self, model_name) -> None:
        try:
            self.__model_name = model_name
            self.__model= joblib.load(f'../Storage/Models/{model_name}.pkl')
        except:
            raise(f"ERROR: The model ({model_name}) doesn't exist")

    def predict_sentiment(self, sentences):
        """
            Predict sentiment using a given model,
            It could be used for single sentiment and bulk as well.

            Returns:
            {
                'Sentiment': str,
                'Positive_prob': float,
                'Negative_prob': float
            }
        """
        if(isinstance(sentences, str)):
            sentences = [sentences]
        try:
            preds = self.__model.predict(sentences)
            preds_proba = self.__model.predict_proba(sentences)
            lens = len(sentences)
            results = []
            for i in range(lens):
                results.append({
                    'Sentiment': preds[i],
                    'Positive_prob': preds_proba[i][1],
                    'Negative_prob': preds_proba[i][0]
                })
            return results if lens > 1 else results[0]
        except Exception as e:
            print("ERROR:", e)

    def retrain(self):
        """
            Re-train a given model again,

            We can use this in case that data records are changed 
            but still in the same format
        """
        try:
            # loading data
            DATA_PATH = '../Storage/Data/External'
            X_train = pd.read_csv(f'{DATA_PATH}/X_train.csv')
            y_train = pd.read_csv(f'{DATA_PATH}/y_train.csv')
            X_test = pd.read_csv(f'{DATA_PATH}/X_test.csv')
            y_test = pd.read_csv(f'{DATA_PATH}/y_test.csv')

            # re-fit the model
            self.__model.fit(X_train, y_train)

            # Replace the new trained model with the stored one
            joblib.dump(self.__model, f'../Storage/Models/{self.__model_name}.pkl')

            # Show very small summary
            print("Training Score: ", self.__model.score(X_train, y_train))
            print("Testing Score: ", self.__model.score(X_test, y_test))
        except Exception as e:
            print("ERROR:", e)

    def evaluate(self):
        """
            Evaluate a trained model performance on test set,

            Showing confusion_matrix, precision, recall and F1-Score in console
        """
        try:
            # loading data
            DATA_PATH = '../Storage/Data/External'
            X_test = pd.read_csv(f'{DATA_PATH}/X_test.csv')
            y_test = pd.read_csv(f'{DATA_PATH}/y_test.csv')

            # get X_test predictions
            y_preds = self.__model.predict(X_test)

            # Show Evaluation Summary
            print(confusion_matrix(y_test, y_preds))
            print("\n===============\n")
            print(classification_report(y_test, y_preds))

        except Exception as e:
            print("ERROR:", e)
