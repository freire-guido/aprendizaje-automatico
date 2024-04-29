# Ejercicio 1
## Separación de datos

Separamos el 20% de los datos para test, porque queremos una buena estimación del AUCROC para el modelo final. Para esto implementamos la función
```desarrollo_evaluacion```, que toma las últimas filas del dataframe. Buscamos que el conjunto de testeo sea una buena representación del dataset,
entonces antes de separar los datos permutamos las filas del dataframe de forma aleatoria usando la función ```sample``` de pandas (toma una muestra
de todo el dataset, sin reposicion. Es decir, lo permuta).

# Ejercicio 2
## Construcción de modelos

Entrenamos el árbol de decisión de altura máxima 3 y estimamos con K-fold cross validation usando ```model_selection.cross_validate``` de sklearn con
cv = 5. Para calcular el AUPRC usamos ```average_precision```, que calcula el área debajo de la curva PRC **sin interpolar**. Osea, algo más bien parecido
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

