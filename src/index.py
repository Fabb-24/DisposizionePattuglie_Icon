from cleanDataset import cleanChicagoAreas, cleanChicagoCrimes
from PrologKB import KB
from patrolArrangement import PatrolArrangement as PA
from icon.learning import SupervisedLearning as SL
from util import printSolution, getBasePath
import os


if __name__ == "__main__":
    # Pulizia dei dataset
    chicagoAreasDf = cleanChicagoAreas()
    chicagoCrimesDf = cleanChicagoCrimes()

    # Apprendimento supervisionato per la previsione della gravità dei crimini
    #supervisedLearning = SL(chicagoCrimesDf, "Severity")
    # Se come parametro si passa il json con i parametri verrano usati invece che cercati
    #res = supervisedLearning.trainModel(os.path.join(getBasePath(), "learning"))
    #print(res)
    

    # Creazione della knowledge base
    kb = KB(chicagoAreasDf, os.path.join(getBasePath(), "learning", "models", "DecisionTree.pkl"), 32, 2, 1)

    pa = PA(kb)
    sol = pa.findBestArrangement()
    printSolution(sol)