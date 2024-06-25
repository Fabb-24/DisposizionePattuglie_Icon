from pyswip import Prolog
from shapely.geometry import MultiPolygon
import pandas as pd
import joblib
from util import removeDuplicates, getBasePath
import os


class KB:
    """
    Classe che rappresenta la base di conoscenza in prolog e fornisce i metodi per manipolarla e interrogarla

    Attributi:
        areasGdf (GeoDataFrame): un GeoDataFrame contenente le aree di Chicago con il loro perimetro
        modelPath (String): Il percorso del modello di machine learning addestrato per la previsione della gravità dei crimini
        week (Int): la settimana in cui si vuole effettuare la previsione
        day (Int): il giorno della settimana in cui si vuole effettuare la previsione
        timeslot (int): la fascia oraria in cui si vuole effettuare la previsione
    """
    
    def __init__(self, areasGdf, modelPath, week, day, timeslot):
        """
        Costruttore della classe. Inizializza la base di conoscenza in prolog

        Parametri:
            areasGdf (GeoDataFrame): un GeoDataFrame contenente le aree di Chicago con il loro perimetro
            modelPath (String): Il percorso del modello di machine learning addestrato per la previsione della gravità dei crimini
            week (Int): la settimana in cui si vuole effettuare la previsione
            day (Int): il giorno della settimana in cui si vuole effettuare la previsione
            timeslot (int): la fascia oraria in cui si vuole effettuare la previsione
        """

        self.prolog = Prolog()
        self.areasGdf = areasGdf
        self.modelPath = modelPath
        self.week = week
        self.day = day
        self.timeslot = timeslot
        self.initializaKB()
    

    def initializaKB(self):
        """
        Metodo che inizializza la base di conoscenza in prolog
        """

        kb = str(os.path.join(getBasePath(), "src", "kb.pl")).replace("\\", "/")
        self.prolog.consult(kb)
        # Definizione dei fatti area per le aree di Chicago
        self.defineAreas()
        # Definizione dei fatti nearAreas per le aree vicine
        self.defineNearAreas()
        # Definizione dei fatti relativi alla gravità delle aree
        self.defineAreaSeverities()
        # Definizione dei fatti relativi alle dimensioni delle aree
        self.defineAreasSize()

    
    def defineAreas(self):
        """
        Metodo che definisce i fatti area per le aree di Chicago, della forma area(AreaNumber)
        """

        for area in self.areasGdf.itertuples():
            self.prolog.assertz(f"area({area.AreaNumber})")


    def defineNearAreas(self):
        """
        Metodo che definisce i fatti nearAreas per le aree vicine, della forma nearAreas(area(A), L).
        Due aree sono considerate vicine se hanno due punti del loro perimetro in comune
        """

        i = 0
        y = 0
        for area1 in self.areasGdf.itertuples():
            perimeter1 = MultiPolygon(area1.Perimeter)
            y = 0
            list = []
            for area2 in self.areasGdf.itertuples():
                if i != y:
                    perimeter2 = MultiPolygon(area2.Perimeter)
                    if perimeter1.touches(perimeter2):
                        list.append(area2.AreaNumber)
                y += 1
            self.prolog.assertz(f"nearAreas(area({area1.AreaNumber}), {list})")
            i += 1


    def defineAreaSeverities(self):
        """
        Metodo che definisce i fatti areaSeverity per le aree di Chicago, della forma severity(area(A), Sev).
        La gravità dei crimini è prevista utilizzando il modello di machine learning addestrato
        """

        loadedModel = joblib.load(self.modelPath)
        for area in self.areasGdf.itertuples():
            data = {
                'Week': self.week,
                'Day': self.day,
                'Time Slot': self.timeslot,
                'Community Area': area.AreaNumber
            }
            prediction = loadedModel.predict(pd.DataFrame(data, index=[0]))
            self.prolog.assertz(f"severity(area({area.AreaNumber}), {int(prediction[0])})")


    def defineAreasSize(self):
        """
        Metodo che definisce i fatti size per le aree di Chicago, della forma size(area(A), S).
        Rappresentano la grandezza delle zone di Chicago
        """

        for area in self.areasGdf.itertuples():
            self.prolog.assertz(f"size(area({area.AreaNumber}), {area.AreaSize})")


    def setAreaPatrol(self, areaNum, patrol):
        """
        Metodo che imposta l'area come pattugliata.
        Aggiunge il fatto patrolArea per l'area specificata, della forma patrolArea(area(A))

        Parametri:
            areaNum (Int): il numero dell'area da impostare come pattugliata
            patrol (Bool): True se l'area è pattugliata, False altrimenti
        """

        # controllo se esiste già un fatto patrolArea per l'area specificata
        res = list(self.prolog.query(f"patrolArea(area({areaNum}), _)"))
        if res:
            self.prolog.retract(f"patrolArea(area({areaNum}), _)")
        self.prolog.assertz(f"patrolArea(area({areaNum}), {'true' if patrol else 'false'})")

    

    def removeAreaPatrol(self, areaNum):
        """
        Metodo che rimuove l'area dalla lista delle aree pattugliate.
        Rimuove il fatto patrolArea per l'area specificata

        Parametri:
            areaNum (Int): il numero dell'area da rimuovere dalla lista delle aree pattugliate
        """

        if list(self.prolog.query(f"patrolArea(area({areaNum}), _)")):
            self.prolog.retract(f"patrolArea(area({areaNum}), _)")


    def isAreaSafe(self, areaNum):
        """
        Metodo che verifica se l'area specificata è sicura

        Parametri:
            areaNum (Int): il numero dell'area da verificare

        Returns:
            List: la lista delle soluzioni trovate dalla query isSafe(area(AreaNumber))
        """

        res = list(self.prolog.query(f"isSafe(area({areaNum}))"))
        return res
    

    def getAreasList(self):
        """
        Metodo che restituisce la lista delle aree di Chicago

        Returns:
            List: la lista delle aree di Chicago
        """

        areasList = []
        res = list(self.prolog.query("area(X)"))
        for area in res:
            areasList.append(area['X'])
        return areasList
    

    def getAreasByDistance(self, areaNum, distance):
        """
        Metodo che restituisce la lista delle aree di Chicago a una distanza specificata dall'area specificata

        Parametri:
            areaNum (Int): il numero dell'area di partenza
            distance (Int): la distanza dall'area di partenza

        Returns:
            List: la lista delle aree a distanza distance dall'area specificata
        """

        areas = list(self.prolog.query(f"distance(area({areaNum}), area(X), {distance})"))
        areasList = []
        for area in areas:
            areasList.append(area['X'])
        return removeDuplicates(areasList)
    

    def getAreaSeverity(self, areaNum):
        """
        Metodo che restituisce la gravità dei crimini dell'area specificata

        Parametri:
            areaNum (Int): il numero dell'area

        Returns:
            Int: la gravità dei crimini dell'area specificata
        """

        severity = list(self.prolog.query(f"severity(area({areaNum}), X)"))
        return severity[0]['X']
    

    def evaluableAreas(self, areas):
        """
        Metodo che restituisce la lista delle aree valutabili

        Parametri:
            areas (List): la lista delle aree da cui selezionare le aree valutabili

        Returns:
            List: la lista delle aree valutabili
        """

        evAreas = list(self.prolog.query("isConsiderable(area(X))"))
        areasList = []
        for evArea in evAreas:
            if evArea['X'] in areas:
                areasList.append(evArea['X'])
        return areasList