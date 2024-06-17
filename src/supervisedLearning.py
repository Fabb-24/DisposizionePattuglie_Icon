import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score, learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import json


'''
Classe che rappresenta il problema di apprendimento supervisionato
'''
class SupervisedLearning:
    '''
    Costruttore della classe. Inizializza i modelli vuoti, i parametri da testare per i modelli, il preprocessor e le metriche di valutazione
    Parametri:
    - dataset (DataFrame): il dataset su cui effettuare l'apprendimento supervisionato
    - target (String): il target del dataset
    '''
    def __init__(self, dataset, target):
        self.dataset = dataset
        self.target = target

        self.empty_models = {
            'DecisionTree': DecisionTreeClassifier(),
            'RandomForest': RandomForestClassifier(),
            'LogisticRegression': LogisticRegression()
        }

        self.param_grids = {
            'DecisionTree': {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]},
            'RandomForest': {'n_estimators': [100, 200], 'max_depth': [10, 20], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2, 4], 'bootstrap': [True]},
            'LogisticRegression': {'C': [0.01, 0.1, 1], 'max_iter': [1000, 2000]}
        }

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('scaler', StandardScaler(), pd.DataFrame(dataset.drop(target, axis=1)).columns)
            ]
        )

        self.scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    

    '''
    Funzione di oversampling per bilanciare il dataset
    Parametri:
    - dataset (DataFrame): il dataset da bilanciare
    - target (String): il target del dataset
    '''
    def oversamplimg(self, dataset, target):
        X = dataset.drop(target, axis=1)
        y = dataset[target]
        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X, y)
        dataset_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        dataset_resampled[target] = y_resampled
        return dataset_resampled

    
    '''
    Funzione che restituisce i migliori parametri per i modelli
    Parametri:
    - X_train (DataFrame): il dataset di training senza il target
    - y_train (DataFrame): il target del dataset di training
    '''
    def bestParams(self, X_train, y_train):
        values = {}
        for model_name, model in self.empty_models.items():
            print(f"\nSearching best params for {model_name} model...")
            grid_search = GridSearchCV(estimator=model, param_grid=self.param_grids[model_name], cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error', refit='neg_mean_squared_error')
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('model', grid_search)
            ])
            pipeline.fit(X_train, y_train)
            values[model_name] = grid_search.best_params_
        return values
    

    '''
    Funzione che addestra i modelli e li salva su file.
    Effettua l'oversampling del dataset, divide il dataset in training e test set, cerca i migliori parametri per i modelli e addestra i modelli tramite k-fold cross-validation.
    I modelli addestrati vengono salvati su file.
    '''
    def trainModel(self, bestParamsFile=None):
        # Oversampling del dataset
        dataset = self.oversamplimg(self.dataset, self.target)

        # Divisione del dataset in training e test set
        X = dataset.drop(self.target, axis=1)
        y = dataset[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        if bestParamsFile is None:
            # Ricerca dei migliori parametri per i modelli
            best_params = self.bestParams(X_train, y_train)
            # Salvataggio dei migliori parametri su file
            with open('models/best_params.json', 'w') as file:
                json.dump(best_params, file)
        else:
            # Caricamento dei migliori parametri da file
            with open(bestParamsFile, 'r') as file:
                best_params = json.load(file)

        # Impostazione dei modelli con i migliori parametri
        models = {
                'DecisionTree': DecisionTreeClassifier(
                    max_depth=best_params['DecisionTree']['max_depth'],
                    min_samples_split=best_params['DecisionTree']['min_samples_split'],
                    min_samples_leaf=best_params['DecisionTree']['min_samples_leaf']
                ),
                'RandomForest': RandomForestClassifier(
                    n_estimators=best_params['RandomForest']['n_estimators'],
                    max_depth=best_params['RandomForest']['max_depth'],
                    min_samples_split=best_params['RandomForest']['min_samples_split'],
                    min_samples_leaf=best_params['RandomForest']['min_samples_leaf'],
                    bootstrap=best_params['RandomForest']['bootstrap']
                ),
                'LogisticRegression': LogisticRegression(
                    C=best_params['LogisticRegression']['C'],
                    max_iter=best_params['LogisticRegression']['max_iter']
                )
            }
        
        cv = RepeatedKFold(n_splits=5, n_repeats=5)

        # Addestramento dei modelli: valutazione e salvataggio su file
        for model_name, model in models.items():
            print(f"\n\nTraining {model_name} model...")
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('model', model)
            ])

            # Valutazione del modello tramite k-fold cross-validation
            for score in self.scoring:
                scores = cross_val_score(pipeline, X, y, cv=cv, scoring=score, n_jobs=-1)
                mean_score = np.mean(scores)
                print(f"{score}: ", mean_score)
            pipeline.fit(X_train, y_train)
            
            # Salvataggio del modello su file
            model_filename = f"models/{model_name}_model.pkl"
            joblib.dump(pipeline, model_filename)
            print(f"Saved {model_name} model to {model_filename}")
        
        # Generazione delle learning curves per i modelli
        for model_name, model in models.items():
            print(f"Generating {model_name} learning curve...")
            self.learningCurve(model, X, y, model_name)

    
    '''
    Funzione che genera la learning curve per un modello.
    Genera la learning curve per il modello passato come parametro e salva il grafico su file.
    Il grafico mostra l'andamento dell'accuracy del modello in funzione del numero di esempi di training.
    Parametri:
    - model (Model): il modello per cui generare la learning curve
    - X (DataFrame): il dataset senza il target
    - y (DataFrame): il target del dataset
    - model_name (String): il nome del modello
    '''
    def learningCurve(self, model, X, y, model_name):
        # Generazione della learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            estimator=model,
            X=X,
            y=y,
            cv=5,
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy'
        )

        # Calcolo delle medie e delle deviazioni standard dei punteggi di training e test
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        # Creazione del grafico della learning curve
        plt.figure()
        plt.title(f"Learning Curve for {model_name}")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.ylim((0.0, 1.1))
        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r", label="Training score")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g", label="Test score")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="red", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="green", label="Cross-validation score")
        plt.legend(loc="best")
        # Salvataggio del grafico
        plt.savefig(f"learningCurves/{model_name}_learningCurve.png")
        plt.close()