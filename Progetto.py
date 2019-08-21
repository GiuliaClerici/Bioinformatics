from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score, roc_curve, confusion_matrix, auc
import sklearn
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# PROGETTO DEL CORSO DI BIOINFORMATICA
# CLERICI GIULIA Matr.910663
# Predizione di varianti genetiche patogeniche di malattie Mendeliane in
# regioni non codificanti del genoma umano con metodi di deep learning

# definizione della funzione strip_first_col che viene utilizzata in seguito per permettere
# l'esclusione della prima colonna dalla lettura del file caricato tramite np.loadtext
def strip_first_col(fname, delimiter=None):
    # apertura del file fname in modalità lettura
    with open(fname, 'r') as fin:
        # per ogni riga del file
        for line in fin:
            try:
                # suddivide la riga, escludendo la prima colonna, e la restituisce
               yield line.split(delimiter, 1)[1]
            except IndexError:
               continue



# caricamento del training set
X = np.loadtxt(strip_first_col('Mendelian.train.tsv'), delimiter="\t", skiprows=1)
# calcolo le dimensioni del training set
size_data = X.shape[0]
# dichiaro e imposto la variabile per le etichette
Y = np.zeros(size_data)
# imposto a 1 le prime 356 etichette ad indicare che i primi 356 campioni sono positivi
P = 356
Y[0:P] = 1

# impostazioni dei parametri per la creazione delle partizioni del training set
# numero esempi positivi del training set originale = P
P = 356
# numero esempi negativi nel training set originale = N
N = 981032
# numero di partizioni = n
n = 100
# fattore di sovraccampionamento della minority class (esempi positivi) = f
f = 2
# numero dei k nearest neighbor usati nell'algoritmo di sovraccampionamento SMOTE = k
k = 5
# fattore di sottocampionamento della majority class (esempi negativi) = m
m = 4
# numero di esempi positivi che vorrò avere nel mio nuovo dataset = pes
pes = (1 + f) * P

# set di esempi positivi
p_set = X[0:P]

# creazione del modello della NN
# modello MLP con layer interamente connessi
model = Sequential()
# costituito da un input layer con 26 neuroni, che prevede dati in input con 26 attributi e come funzione di attivazione una sigmoide
model.add(Dense(26, input_dim=26, activation='sigmoid'))
# dichiaro una variabile flag atta alla selezione del numero di neuroni presente nell'hidden layer
flag = 0
# se flag è impostata a 0
if flag == 0:
    # imposto il numero di neuroni pari a 6
    num_neurons = 6
# qualora flag sia impostata ad un numero diverso da 0
else:
    # il numero di neuroni è impostato a 13
    num_neurons = 13
# hidden layer con un numero di neuroni pari a num_neurons e come funzione di attivazione una sigmoide
model.add(Dense(num_neurons, activation='sigmoid'))
# output layer con 1 neurone, indicante la classe predetta, e come funzione di attivazione una sigmoide
model.add(Dense(1, activation='sigmoid'))
# stampa della tabella riassuntiva dell'architettura del modello
print(model.summary())
# compilazione del modello
# avente come funzione di perdita la Mean Squared Error, come ottimizzazione il criterio di discesa stocastica del gradiente
model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

# partizionamento di N in n partizioni che non presentano sovrapposizioni
# dichiaro i la variabile che conteggia i cicli
i = 1
# dichiaro step come la quantità di esempi negativi (9810) del training set che andranno a
# comporre la nuova partizione che fungerà da training set per il modello,
# ad eccezione della 100esima partizione che includerà i restanti campioni (9842)
step = np.floor_divide(N, n)
# al fine di partizionare gli esempi negativi, escludo quelli positivi partendo dal 357esimo campione, all'indice 356
start = 356
# ciclo while in cui viene creato, ad ogni ciclo, un diverso training set
# da sottoporre al modello. Il nuovo set di training sarà composto da tutti gli esempi positivi
# del training set originale, una partizione degli esempi negativi e degli ulteriori esempi positivi
# artificiali generati dall'algoritmo SMOTE, che si occupa di sovraccampionare i campioni
# della minority class al fine di bilanciare il training set
while i <= n:
    # definisco l'indice di stop che delimita la quantità di esempi negativi considerati
    stop = start + step
    # qualora la partizione sia la 100esima
    if i == 100:
        # considero tutti gli esempi negativi restanti in quanto non riuscirebbero a formare
        # una partizione considerevole in quanto di dimensioni minori del passo 'step' dichiarato
        n_set = X[start:]
    # qualora sia un'ennesima partizione con n!=100
    else:
        # seleziono di volta in volta una partizione degli esempi negativi
        # in modo tale che le diverse partizioni non diano sovrapposizioni
        # selezionando quindi esempi negativi sempre diversi
        n_set = X[start:stop]
    # cambio l'indice di start della selezione della partizione di esempi negativi per il prossimo ciclo
    start = stop
    # creo un set in cui sono presenti gli esempi positivi e a seguito la partizione di esempi negativi selezionata
    X_set = np.concatenate((p_set, n_set), axis=0)
    # calcolo la dimensione del set
    rows = X_set.shape[0]
    # creo la variabile contenente le etichette
    Y_set = np.zeros(rows)
    # impostando le prime 356 etichette a 1 in quanto relative agli esempi positivi
    Y_set[0:P] = 1
    # dichiarazione dell'algoritmo SMOTE atto all'oversampling della minority class,
    # ossia della classe degli esempi positivi, al fine di produrre un training set bilanciato
    p_smote = SMOTE(ratio='minority')
    # creazione di un nuovo set, con relative etichette, bilanciato dalla presenza di nuovi esempi positivi
    # artificiali generati dall'applicazione dell'algoritmo SMOTE
    X_SMOTE, Y_SMOTE = p_smote.fit_sample(X_set, Y_set)
    # FASE DI TRAINING
    # fase di apprendimento del modello
    # utilizzo il nuovo set bilanciato per la fase di apprendimento del modello
    # impostando i valori relativi alle epoche e alla batch_size
    model.fit(X_SMOTE, Y_SMOTE, epochs=100, batch_size=10)
    # valutazione del modello sul set utilizzato per il training
    print("Valutazione del modello sul training set.")
    # calcolo degli scores, ossia dei valori delle metriche
    scores = model.evaluate(X_SMOTE, Y_SMOTE)
    # stampa dei valori delle metriche calcolati
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    # procedo con il ciclo while
    i = i + 1

# salvataggio del modello
model_json = model.to_json()
with open("model_100epochs_6neur.json", "w") as json_file:
    json_file.write(model_json)
# salvataggio dei pesi della rete
model.save_weights("model_100epochs_6neur.h5")
print("Modello salvato.")

# generazione delle etichette di output del modello, dato in ingresso il training set originale
# y_scores = predizioni effettuate dal modello
y_scores = model.predict_classes(X)
# calcolo dell'AUROC - Area under the Receiver Operating Characteristic,
# presi in considerazione le etichette giuste e le predizioni effettuate
auroc_train = sklearn.metrics.roc_auc_score(Y, y_scores)
print("Valore di Auroc sul training set:", auroc_train)
# calcolo del valore medio della precisione confrontando le etichette giuste con quelle predette dal modello
avg_precision_training = sklearn.metrics.average_precision_score(Y, y_scores)
print("Precisione media: ", avg_precision_training)
# calcolo dei valori di precision, recall e thresholds
precision_train, recall_train, thresholds_train = precision_recall_curve(Y, y_scores)
# stampa dei valori di precision e recall
print("Precision: ", precision_train, "; Recall: ", recall_train)

# metriche di valutazione per il training set
# stampa del grafico AUPRC per la valutazione sul training set
print("Grafico auprc:")
plt.step(recall_train, precision_train, color='b', alpha=0.2, where='post')
plt.fill_between(recall_train, precision_train, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('AUPRC - Training: AP={0:0.2f}'.format(sklearn.metrics.average_precision_score(Y, y_scores)))
plt.show()

# stampa del grafico AUROC per la valutazione sul training set
# false positive rate
falsepr = dict()
# true positive rate
truepr = dict()
# valore AUROC
roc_auc_train = dict()
# calcolo della curva ROC
falsepr, truepr, _ = roc_curve(Y, y_scores)
# calcolo del valore AUROC
roc_auc_train = auc(falsepr, truepr)
# stampo il grafico AUROC
plt.figure()
lw = 2
plt.plot(falsepr, truepr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_train)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUROC - Training')
plt.legend(loc="lower right")
plt.show()

#FASE DI TEST
# caricamento del test set
X_test = np.loadtxt(strip_first_col('Mendelian.test.tsv'), delimiter="\t", skiprows=1)
# numero di esempi positivi nel test set
P_test = 40
# numero di esempi totali presenti nel test set
size_data_test = X_test.shape[0]
# dichiaro e imposto la variabile per le etichette dei dati di test
Y_test = np.zeros(size_data_test)
# imposto le prime P_test etichette a 1 ad indicare gli esempi positivi, che sono i primi 40 del test set
Y_test[0:P_test] = 1

# FASE DI TEST
print("Valutazione del modello sul test set.")
# valutazione delle prestazioni del modello sul test set
predictions = model.predict_classes(X_test)

# calcolo dell'AUROC sul test set
auroc_test = sklearn.metrics.roc_auc_score(Y_test, predictions)
print("Auroc score di test: ", auroc_test)
# calcolo del valore medio della precisione confrontando le etichette giuste con quelle predette dal modello
avg_precision_test = sklearn.metrics.average_precision_score(Y_test, predictions)
print("Precisione media: ", avg_precision_test)
# calcolo dei valori di precision, recall, tresholds tramite l'AUPRC
precision_test, recall_test, thresholds_test = precision_recall_curve(Y_test, predictions)
# stampa dei valori di precision e recall
print("Precision: ", precision_test, "; Recall: ", recall_test)

# stampa del grafico AUPRC per la valutazione sul test set
print("Grafico auprc:")
plt.step(recall_test, precision_test, color='b', alpha=0.2, where='post')
plt.fill_between(recall_test, precision_test, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('AUPRC - Test: AP={0:0.2f}'.format(sklearn.metrics.average_precision_score(Y_test, predictions)))
plt.show()

# calcolo del grafico AUROC nella valutazione sul test set
# false positive rate
fpr = dict()
# true positive rate
tpr = dict()
# vaariabile che conterrà il valore di AUROC
roc_auc = dict()
# calcolo della curva ROC
fpr, tpr, _ = roc_curve(Y_test, predictions)
# calcolo dell'AUROC
roc_auc = auc(fpr, tpr)
# stampa del grafico AUROC
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUROC - Test')
plt.legend(loc="lower right")
plt.show()

# calcolo della matrice di confusione nella valutazione sul training set
# veri negativi
truen= 0
# falsi positivi
falsep = 0
# falsi negativi
falsen = 0
# veri positivi
truep = 0
# calcolo dei valori della matrice di confusione
truen, falsep, falsen, truep = confusion_matrix(Y, y_scores).ravel()
# stampa dei valori
print("tn, fp, fn, tp per training:", truen, falsep, falsen, truep)

# calcolo della matrice di confusione nella valutazione sul test set
# veri negativi
tn= 0
# falsi positivi
fp = 0
# falsi negativi
fn = 0
# veri positivi
tp = 0
# calcolo dei valori della matrice di confusione
tn, fp, fn, tp = confusion_matrix(Y_test, predictions).ravel()
# stampa dei valori
print("tn, fp, fn, tp per test:", tn, fp, fn, tp)