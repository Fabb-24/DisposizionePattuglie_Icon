# DisposizionePattuglie_Icon

## Librerie Python

Per l'esecuzione del programma è necessario installare delle specifiche libreria python elencate nel file "requirements.txt" presente nel percorso /src.  
Per installare velocemente tutte le librerie occorre aprire una istanza del prompt dei comandi e digitare la seguente stringa:
```
pip install -r <percorso requirements.txt>
```
dove al posto di \<percorso requirements.txt> va inserito il percorso assoluto del file indicato.  

## SWI-Prolog

Il progetto fa uso del linguaggio Prolog. Per poterlo utilizzare occorre scaricare anche il programma SWI-Prolog dalla pagina di download del sito ufficiale:  
`https://www.swi-prolog.org/Download.html`  
  
Se nonostante l'installazione si riscontrano problemi nell'esecuzione del programma dovuti a SWI-Prolog si consiglia di provare a installare una versione meno recente.  

## Esecuzione del programma

Per eseguire il programma bisogna eseguire il file "index.py" presente nel percorso /src.  
  
Verrà chiesto inizialmente se si vuole eseguire l'apprendimento supervisionato o utilizzare dei modelli già addestrati nel caso fossero presenti. Se non sono presenti modelli nell'apposito percorso si è obbligati a crearne facendo l'apprendimento supervisionato.  
Dopo verrà richiesto il modello da utilizzare per le predizioni, facendo scegliere tra quelli disponibili. In più verranno chieste informazioni riguardanti la settimana, il giorno e la fascia oraria.  
Una volta inserite le informazioni necessarie, inizierà la risoluzione del problema di ottimizzazione che darà come risultato la lista delle aree a cui assegnare la pattuglia.  
La durata del problema di ottimizzazione è variabile e dipende dalle informazioni inserite.