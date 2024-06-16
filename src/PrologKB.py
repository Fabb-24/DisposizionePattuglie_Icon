from pyswip import Prolog
from shapely.geometry import MultiPolygon
import pandas as pd
import joblib
from util import removeDuplicates

'''
Classe che rappresenta la base di conoscenza in prolog e fornisce i metodi per manipolarla e interrogarla
'''
class KB:
    '''
    Costruttore della classe. Inizializza la base di conoscenza in prolog
    Parametri:
    - areasGdf (GeoDataFrame): un GeoDataFrame contenente le aree di Chicago con il loro perimetro
    - modelName (String): il nome del modello di machine learning da utilizzare per la previsione della gravità dei crimini (RandomForest, DecisionTree, LogisticRegression)
    - week (Int): la settimana in cui si vuole effettuare la previsione
    - day (Int): il giorno della settimana in cui si vuole effettuare la previsione
    - timeslot (int): la fascia oraria in cui si vuole effettuare la previsione
    '''
    def __init__(self, areasGdf, modelName, week, day, timeslot):
        self.prolog = Prolog()
        self.areasGdf = areasGdf
        self.modelName = modelName
        self.week = week
        self.day = day
        self.timeslot = timeslot
        self.initializaKB()
    

    '''
    Metodo che inizializza la base di conoscenza in prolog
    '''
    def initializaKB(self):
        # Definizione del fatto patrolArea per la zona 78 (inesistente) per evitare errori di esistenza in caso di query
        self.prolog.assertz("patrolArea(78)")
        # Definizione dei fatti nearAreas per le aree vicine
        self.defineNearAreas()
        # Definizione delle regole relative alla gravità delle aree
        self.defineAreaSeverityRules()
        # Definizione delle regole relative alla distanza tra le aree (distanza 1 e 2)
        self.prolog.assertz("distance(X, Y, D) :- nearAreas(X, Y), D is 1")
        self.prolog.assertz("distance(X, Y, D) :- nearAreas(X, Y), D is 2")
        self.prolog.assertz("distance(X, Y, D) :- nearAreas(X, Z), nearAreas(Z, Y), D is 2, X \= Y, \+ nearAreas(X, Y)")
        # Definizione delle regole relative alla sicurezza delle aree
        self.prolog.assertz("secureArea(X) :- patrolArea(X)")
        self.prolog.assertz("secureArea(X) :- areaSeverity(X, D), D is 1, distance(X, Y, 1), patrolArea(Y)")
        self.prolog.assertz("secureArea(X) :- areaSeverity(X, D), D is 0, distance(X, Y, 2), patrolArea(Y)")


    '''
    Metodo che definisce i fatti nearAreas per le aree vicine, della forma nearAreas(AreaNumber1, AreaNumber2).
    Due aree sono considerate vicine se hanno due punti del loro perimetro in comune
    '''
    def defineNearAreas(self):
        i = 0
        y = 0
        for area1 in self.areasGdf.itertuples():
            perimeter1 = MultiPolygon(area1.Perimeter)
            y = 0
            for area2 in self.areasGdf.itertuples():
                if i != y:
                    perimeter2 = MultiPolygon(area2.Perimeter)
                    if perimeter1.touches(perimeter2):
                        self.prolog.assertz(f"nearAreas({area1.AreaNumber}, {area2.AreaNumber})")
                y += 1
            i += 1
    

    '''
    Metodo che definisce i fatti areaSeverity per le aree di Chicago, della forma areaSeverity(AreaNumber, Severity).
    La gravità dei crimini è prevista utilizzando il modello di machine learning addestrato
    '''
    def defineAreaSeverityRules(self):
        modelFile = f"models/{self.modelName}_model.pkl"
        loadedModel = joblib.load(modelFile)

        for area in self.areasGdf.itertuples():
            data = {
                'Week': self.week,
                'Day': self.day,
                'Time Slot': self.timeslot,
                'Community Area': area.AreaNumber
            }
            prediction = loadedModel.predict(pd.DataFrame(data, index=[0]))
            self.prolog.assertz(f"areaSeverity({area.AreaNumber}, {int(prediction[0])})")


    '''
    Metodo che imposta l'area come pattugliata. Aggiunge il fatto patrolArea per l'area specificata, della forma patrolArea(AreaNumber)
    Parametri:
    - areaNum (Int): il numero dell'area da impostare come pattugliata
    '''
    def setAreaPatrol(self, areaNum):
        self.prolog.assertz(f"patrolArea({areaNum})")
    

    '''
    Metodo che rimuove l'area dalla lista delle aree pattugliate. Rimuove il fatto patrolArea per l'area specificata
    Parametri:
    - areaNum (Int): il numero dell'area da rimuovere dalla lista delle aree pattugliate
    '''
    def removeAreaPatrol(self, areaNum):
        if self.isAreaPatrolled(areaNum) > 0:
            self.prolog.retract(f"patrolArea({areaNum})")
    

    '''
    Metodo che verifica se l'area specificata è pattugliata
    Parametri:
    - areaNum (Int): il numero dell'area da verificare
    '''
    def isAreaPatrolled(self, areaNum):
        return len(list(self.prolog.query(f"patrolArea({areaNum})"))) > 0


    '''
    Metodo che verifica se l'area specificata è sicura
    Parametri:
    - areaNum (Int): il numero dell'area da verificare
    '''
    def isAreaSecure(self, areaNum):
        res = list(self.prolog.query(f"secureArea({areaNum})"))
        if res == []:
            return False
        return True
    

    '''
    Metodo che restituisce la lista di tutte le aree di Chicago
    '''
    def getAreasList(self):
        areasList = []
        for area in self.areasGdf.itertuples():
            areasList.append(area.AreaNumber)
        return areasList
    

    '''
    Metodo che restituisce la lista delle aree di Chicago con la gravità specificata
    Parametri:
    - gravity (Int): la gravità dei crimini
    '''
    def getAreasByGravity(self, gravity):
        areas = list(self.prolog.query(f"areaSeverity(X, {gravity})"))
        areasList = []
        for area in areas:
            areasList.append(area['X'])
        return areasList
    

    '''
    Metodo che restituisce la lista delle aree di Chicago a una distanza specificata dall'area specificata
    Parametri:
    - areaNum (Int): il numero dell'area di partenza
    - distance (Int): la distanza dall'area di partenza
    '''
    def getAreasByDistance(self, areaNum, distance):
        areas = list(self.prolog.query(f"distance({areaNum}, X, {distance})"))
        areasList = []
        for area in areas:
            areasList.append(area['X'])
        return removeDuplicates(areasList)
    

    '''
    Metodo che restituisce la gravità dei crimini dell'area specificata
    Parametri:
    - areaNum (Int): il numero dell'area
    '''
    def getAreaGravity(self, areaNum):
        severity = list(self.prolog.query(f"areaSeverity({areaNum}, X)"))
        return severity[0]['X']