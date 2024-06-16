'''
Funzione che rimuove i duplicati in una lista
Parametri:
- lst (List): la lista da cui rimuovere i duplicati
'''
def removeDuplicates(lst):
    return list(dict.fromkeys(lst))