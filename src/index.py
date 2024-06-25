from cleanDataset import cleanChicagoAreas, cleanChicagoCrimes
from PrologKB import KB
from patrolArrangement import PatrolArrangement as PA
from icon.learning import SupervisedLearning as SL
from util import printSolution, printResults, getBasePath
import os


if __name__ == "__main__":
    # Percorso contenente i modelli di apprendimento utilizzabili
    modelsPath = os.path.join(getBasePath(), "learning", "models")
    models = os.listdir(modelsPath)

    # Pulizia del dataset delle aree di Chicago
    chicagoAreasDf = cleanChicagoAreas()

    # Scelta tra apprendimento supervisionato e utilizzo di un modello già addestrato
    while True:
        print("Scelta:\n1. Eseguire l'apprendimento supervisionato\n2. Utilizzare un modello già addestrato precedentemente\n")
        choice = int(input("Inserire la scelta: "))
        if choice != 1 and choice != 2:
            print("Scelta non valida\n")
        elif choice == 2 and len(models) == 0:
            print("Non sono presenti modelli di apprendimento precedentemente addestrati\n")
        else:
            break
    
    if choice == 1:
        print("\n--- Apprendimento supervisionato ---\n\n")
        # Pulizia del dataset dei crimini di Chicago
        chicagoCrimesDf = cleanChicagoCrimes()
        # Apprendimento supervisionato per la previsione della gravità dei crimini
        supervisedLearning = SL(chicagoCrimesDf, "Severity")
        # Se come parametro si passa il json con i parametri verrano usati invece che cercati
        res = supervisedLearning.trainModel(os.path.join(getBasePath(), "learning"))
        printResults(res)
        print("\nApprendimento supervisionato completato")
    
    # Disposizione delle pattuglie
    print("\n\n--- Ricerca della disposizione migliore delle pattuglie ---\n\n")
    # scelta del modello di apprendimento da utilizzare. Vengono proposti i modelli disponibili nel percorso /learning/models
    while True:
        print("Scegli il modello di apprendimento da utilizzare:")
        modelsPath = os.path.join(getBasePath(), "learning", "models")
        models = os.listdir(modelsPath)
        for i, model in enumerate(models):
            print(f"{i+1}. {model}")
        print("\nModello scelto:", end=" ")
        choice = int(input())
        if choice < 1 or choice > len(models):
            print("Scelta non valida\n")
        else:
            break

    print("\nInserire il numero della settimana (1-52):", end=" ")
    week = int(input())
    print("Inserire il giorno della settimana (1-7): ", end=" ")
    day = int(input())
    print("Inserire l'ora del giorno (0-23): ", end=" ")
    hour = int(input())
    print("\n")
    # Creazione della base di conoscenza
    kb = KB(chicagoAreasDf, os.path.join(getBasePath(), "learning", "models", "DecisionTree.pkl"), week, day, hour)
    # Problema di ottimizzazione per la ricerca della disposizione
    pa = PA(kb)
    sol = pa.findBestArrangement()
    printSolution(sol)