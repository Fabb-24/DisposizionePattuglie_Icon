from cleanDataset import cleanChicagoAreas, cleanChicagoCrimes
from PrologKB import KB
from patrolArrangement import PatrolArrangement as PA
from icon.learning import SupervisedLearning as SL
from util import printSolution, printResults, getBasePath
import os


if __name__ == "__main__":
    # Pulizia dei dataset
    chicagoAreasDf = cleanChicagoAreas()
    chicagoCrimesDf = cleanChicagoCrimes()

    # Decommentare per eseguire l'apprendimento supervisionato
    '''# Apprendimento supervisionato per la previsione della gravità dei crimini
    supervisedLearning = SL(chicagoCrimesDf, "Severity")
    # Se come parametro si passa il json con i parametri verrano usati invece che cercati
    res = supervisedLearning.trainModel(os.path.join(getBasePath(), "learning"))
    printResults(res)'''
    

    # Creazione della knowledge base
    week = 32
    day = 2
    hour = 1
    kb = KB(chicagoAreasDf, os.path.join(getBasePath(), "learning", "models", "DecisionTree.pkl"), week, day, hour)

    pa = PA(kb)
    sol = pa.findBestArrangement()
    printSolution(sol)