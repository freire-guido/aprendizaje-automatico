# Ejercicio 1
## Separación de datos

Dado que tenemos una base de datos acotada (500 datos), consideramos que utilizar el 10% de los datos como test set, es poco confiable (50 datos). Consideramos que 50 datos no nos proporcionan realmente una certeza acorde a la que buscamos para testear que tan bueno es nuestro modelo. Con este objetivo, decidimos separar el 20% de los datos para test, porque queremos una buena estimación del AUCROC para el modelo final. Para esto implementamos la función ```desarrollo_evaluacion```, que toma las últimas filas del dataframe. 

Sin embargo, los datos podrían estar "cargados" al dataset de cierta manera provocando asi que los ultimos datos del dataset tengan un patron que el modelo sea incapaz de generalizar. En este sentido buscamos que el conjunto de testeo sea una buena representación del dataset, entonces antes de separar los datos permutamos las filas del dataframe de forma aleatoria usando la función ```sample``` de pandas (toma una muestra de todo el dataset, sin reposicion. Es decir, lo permuta).

Una vez separados los datos en un conjunto de entrenamiento y un conjunto de evaluacion o test, aplicaremos preferentemente K-fold cross validation (a menos que una consigna indique otra forma de entrenamiento) y todos los modelos o metodos sobre el conjunto de entrenamiento sin tocar el conjunto de test, para no obtener ni inferir informacion respecto de este conjunto. Finalmente, al momento de indicar el indice de certeza que tiene nuestro modelo utilizaremos este conjunto de evaluacion, sobre el cual aplicaremos el modelo que consideremos pertinente y del cual, en ningun momento habremos obtenido informacion al respecto, como si predicieramos datos "nuevos" en nuestro data frame.

# Ejercicio 2
## Construcción de modelos

1. Entrenamos el árbol de decisión de altura máxima 3 y estimamos con K-fold cross validation usando ```model_selection.cross_validate``` de sklearn con
cv = 5.

2. Para calcular el AUPRC usamos ```average_precision```, que calcula el área debajo de la curva PRC **sin interpolar**. Osea, algo más bien parecido
a la suma de Riemann:

$$ \text{AP} = \sum_n (R_n - R_{n-1})P_n $$

donde $R_n$ es el recall y $P_n$ la precisión. Los resultados que obtuvimos estan en la tabla:

| Permutación | Accuracy (training) | Accuracy (validación) | AUPRC (training) | AUPRC (validación) | AUCROC (training) | AUCROC (validación) |
| -----       | -----               | -----                 | -----            | -----              | -----             | -----               |
| 1           | 0.821               | 0.687                 | 0.740            | 0.475              | 0.840             | 0.650               |
| 2           | 0.759               | 0.662                 | 0.625            | 0.425              | 0.800             | 0.666               |
| 3           | 0.853               | 0.737                 | 0.737            | 0.507              | 0.835             | 0.737               |
| 4           | 0.840               | 0.650                 | 0.718            | 0.343              | 0.839             | 0.550               |
| 5           | 0.828               | 0.700                 | 0.723            | 0.480              | 0.860             | 0.666               |
| Promedios   | 0.820               | 0.687                 | 0.709            | 0.446              | 0.837             | 0.654               |
| Global      | (NO)                | 0.687                 | (NO)             | 0.397              | (NO)              | 0.619               |

*Tabla 2.1*

3. Definimos una ```ParameterGrid``` de sklearn con los parámetros que queremos probar con el árbol de decisión y usamos ```cross_validate``` con la métrica accuracy. Los resultados fueron:

| Altura máxima | Criterio de corte | Accuracy (training) | Accuracy (validación) |
| -----         | -----             | -----               | -----                 |
| 3             | Gini              | 0.820               | 0.687                 |
| 5             | Gini              | 0.938               | 0.657                 |
| Inf           | Gini              | 1.000               | 0.650                 |
| 3             | Entropía          | 0.795               | 0.717                 |
| 5             | Entropía          | 0.908               | 0.657                 |
| Inf           | Entropía          | 1.000               | 0.655                 |

*Tabla 2.2*

4. De la tabla 2.1 vemos que los folds obtienen scores parecidos dentro de una métrica. Interpretamos que los datos están representados de forma pareja en cada uno de ellos. Tiene sentido que el promedio sea cercano a cada uno de los folds. También cabe notar que usar la evaluación *Global* recomendada en clase es muy cercana al promedio, pero siempre por debajo del mismo. Según 2.1 la peor métrica para este modelo es AUPRC.

En la tabla 2.2 es claro que subir la altura máxima del árbol aumenta la precisión en training. De hecho, al entrenar un árbol completo cada dato de entrenamiento está bien clasificado (precisión 1). Sin embargo la varianza es muy alta ("overfitting") y la precisión empeora. El criterio de corte entropía es marginalmente mejor que gini.

# Ejercicio 3
## Comparacion de algoritmos

Comparemos la certeza de ciertos algoritmos de clasificacion mediante la metrica AUCROC resultante de un 5-fold cross validation. En este caso compararemos los algoritmos:
* Árboles de decisión (ADD)
* KNN (k-vecinos más cercanos)
* SVM (Support vector machine)
* LDA (Linear discriminant analysis)
* Naïve Bayes

Además compararemos "intra" algoritmos, es decir, compararemos mismos algoritmos con distintos hiperparametros y asi poder analizar como afectan el rendimiento de cada algoritmo. En este sentido, utilizaremos un ´´´RandomizedSearchCV´´´ que nos permite buscar de forma aleatoria en la grilla o "hipercubo" que generan los distintos intervalos/regiones para cada hiperparametro de cada modelo.

Comenzaremos delimitando cuales hiperparametros utilizar para cada modelo y sus respectivos intervalos y/o valores que pueden tomar los mismos.

| Modelo        | Hiperparámetro    | Intervalo / Valores posibles      | Descripción           |
| -----         | -----             | -----                             | -----                 |
| ADD           | max_depth         | [1,2, ..., 19]                    | Altura maxima del arbol|
| ADD           | Criterion         | ['gini', 'entropy', 'log_loss']   | Estrategia utilizada para separar en cada nodo|
| ADD           | max_features      | ['sqrt', 'log2']                  | Numero de features al considerar el mejor corte|
| KNN           | n_neighbors       | [1,2, ..., 19]                    | Cantidad de vecinos mas cercanos a observar|
| KNN           | metric            | ['l1', 'l2', 'cosine']            | Metrica utilizada para calcular la distancia entre observaciones|
| SVM           | C                 | [1, 1.5, 2, ...., 19.5]          | Citerio de regularizacion, inversamente proporcional a la fuerza de regularizacion buscada|
| SVM           | tol               | [1e-1, 1e-2, ..., 1e-5]           | Criterio de tolerancia para frenar algoritmo|
| SVM           | kernel            | ['linear', 'rbf']                 | Nucleo utilizado en el algoritmo, "genera" la forma de las regiones delimitadas|
| LDA           | solver            | ['lsqr', 'eigen']                 | Metodo utilizado para resolver el problema|
| LDA           | shrinkage         | [None, 'auto', 0.1, 0.5, 1.0]     | Controla si se utiliza o no el enfoque de "contraccion" de la amtriz de covarianza|
| Naive Bayes   | priors            | [0, 0.01, 0.02, ...., 0.99]       | Probabilidades inciales para las clases|

(continuar)

