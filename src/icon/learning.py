import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score, learning_curve, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import json
import os

class SupervisedLearning:
    """
    Classe che si occupa dell'apprendimento supervisionato attraverso l'utilizzo di modelli di classificazione: DecisionTree, RandomForest e LogisticRegression

    Attributi:
        dataset (DataFrame): il dataset su cui effettuare l'apprendimento supervisionato
        target (String): il target del dataset
    """

    __all__ = ['trainModel']


    def __init__(self, dataset, target):
        """
        Costruttore della classe. Inizializza i modelli vuoti, i parametri da testare per i modelli, il preprocessor e le metriche di valutazione

        Parametri:
            dataset (DataFrame): il dataset su cui effettuare l'apprendimento supervisionato
            target (String): il target del dataset
        """

        self.dataset = dataset
        self.target = target

        self.empty_models = {
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'AdaBoost': AdaBoostClassifier(),
        }

        self.param_grids = {
            'Decision Tree': {'criterion': ['gini', 'entropy'],
                            'max_depth': [10, 20, 30, 40],
                            'min_samples_split': [2, 5, 10, 20],
                            'min_samples_leaf': [1, 2, 5, 10]},

            'Random Forest': {'criterion': ['gini', 'entropy'],
                            'n_estimators': [100, 200],
                            'max_depth': [10, 20],
                            'min_samples_split': [2, 5, 10],
                            'min_samples_leaf': [1, 2, 5],
                            'bootstrap': [True, False]},
                            
            'AdaBoost': {'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 1, 10],
                        'algorithm': ['SAMME']}
        }

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('scaler', StandardScaler(), pd.DataFrame(dataset.drop(target, axis=1)).columns)
            ]
        )

        self.scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    

    def trainModel(self, savePath, bestParamsFile=None):
        """
        Funzione che si occupa dell'addestramento dei modelli di classificazione DecisionTree, RandomForest e LogisticRegression.
        Ogni modello viene valutato secondo le metriche: accuracy, precision, recall e f1.

        Parametri:
            savePath (String): il percorso in cui salvare i migliori parametri, i modelli e le learning curves
            bestParamsFile (String): il file json contenente i migliori parametri per i modelli. Se non specificato, vengono cercati i migliori parametri
        
        Return:
            res (Dict): un dizionario contenente i valori delle metriche per i modelli addestrati
        """

        # Creazione delle cartelle per i modelli e le learning curves
        modelsPath = os.path.join(savePath, 'models')
        learningCurvesPath = os.path.join(savePath, 'learningCurves')
        if not os.path.exists(modelsPath):
            os.makedirs(modelsPath)
        if not os.path.exists(learningCurvesPath):
            os.makedirs(learningCurvesPath)

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
            bpPath = os.path.join(savePath, 'best_params.json')
            with open(bpPath, 'w') as file:
                json.dump(best_params, file)
        else:
            # Caricamento dei migliori parametri da file
            with open(bestParamsFile, 'r') as file:
                best_params = json.load(file)

        # Impostazione dei modelli con i migliori parametri
        models = {
                'DecisionTree': DecisionTreeClassifier(
                    criterion=best_params['Decision Tree']['criterion'],
                    max_depth=best_params['Decision Tree']['max_depth'],
                    min_samples_split=best_params['Decision Tree']['min_samples_split'],
                    min_samples_leaf=best_params['Decision Tree']['min_samples_leaf']
                ),
                'RandomForest': RandomForestClassifier(
                    n_estimators=best_params['Random Forest']['n_estimators'],
                    max_depth=best_params['Random Forest']['max_depth'],
                    min_samples_split=best_params['Random Forest']['min_samples_split'],
                    min_samples_leaf=best_params['Random Forest']['min_samples_leaf'],
                    bootstrap=best_params['Random Forest']['bootstrap'],
                    criterion=best_params['Random Forest']['criterion']
                ),
                'AdaBoost': AdaBoostClassifier(
                    n_estimators=best_params['AdaBoost']['n_estimators'],
                    learning_rate=best_params['AdaBoost']['learning_rate'],
                    algorithm=best_params['AdaBoost']['algorithm']
                )
            }
        
        cv = RepeatedKFold(n_splits=5, n_repeats=5)

        res = {}

        # Addestramento dei modelli: valutazione e salvataggio su file
        for model_name, model in models.items():
            res[model_name] = {}

            print(f"\n\nTraining {model_name} model...")
            pipeline = ImbPipeline([
                ('preprocessor', self.preprocessor),
                ('model', model)
            ])

            # Addestramento e valutazione del modello tramite k-fold cross-validation
            for score in self.scoring:
                scores = cross_val_score(pipeline, X, y, cv=cv, scoring=score, n_jobs=-2)
                mean_score = np.mean(scores)
                res[model_name][score.split("_")[0]] = mean_score
            pipeline.fit(X_train, y_train)
            
            # Salvataggio del modello su file
            modelPath = os.path.join(savePath, 'models', f"{model_name}.pkl")
            joblib.dump(pipeline, modelPath)
            print(f"Saved {model_name} model to {modelPath}")
        
        # Generazione delle learning curves per i modelli
        for model_name, model in models.items():
            print(f"\n\nGenerating {model_name} learning curve...")
            self.learningCurve(model, X, y, model_name, savePath)

        return res
    

    def bestParams(self, X_train, y_train):
        """
        Funzione che restituisce i migliori parametri per i modelli.
        Utilizza la tecnica GridSearchCV per la ricerca dei migliori parametri

        Parametri:
            X_train (DataFrame): il dataset di training senza il target
            y_train (DataFrame): il target del dataset di training
        """

        values = {}
        for model_name, model in self.empty_models.items():
            print(f"\nSearching best params for {model_name} model...")
            grid_search = GridSearchCV(estimator=model, param_grid=self.param_grids[model_name],
                                        cv=5, n_jobs=-1, verbose=2, scoring='f1_macro', refit='f1_macro')
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('model', grid_search)
            ])
            pipeline.fit(X_train, y_train)
            values[model_name] = grid_search.best_params_
        return values
    

    def oversamplimg(self, dataset, target):
        """
        Funzione di oversampling per bilanciare il dataset. Utilizza la tecnica SMOTE per bilanciare il dataset

        Parametri:
            dataset (DataFrame): il dataset da bilanciare
            target (String): il target del dataset
        """

        X = dataset.drop(target, axis=1)
        y = dataset[target]
        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X, y)
        dataset_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        dataset_resampled[target] = y_resampled
        return dataset_resampled


    def learningCurve(self, model, X, y, model_name, savePath):
        """
        Funzione che genera la learning curve per il modello passato come parametro.
        Salva il grafico su file

        Parametri:
            model (Model): il modello per cui generare la learning curve
            X (DataFrame): il dataset senza il target
            y (DataFrame): il target del dataset
            model_name (String): il nome del modello
        """

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
        test_scores_mean = np.mean(test_scores, axis=1)

        # Creazione del grafico della learning curve
        plt.figure()
        plt.title(f"Learning Curve for {model_name}")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.ylim((0.0, 1.1))
        plt.grid()
        plt.plot(train_sizes, train_scores_mean, 'o-', color="red", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="green", label="Test score")
        plt.legend(loc="best")
        # Salvataggio del grafico
        plt.savefig(os.path.join(savePath, 'learningCurves', f"{model_name}.png"))
        plt.close()