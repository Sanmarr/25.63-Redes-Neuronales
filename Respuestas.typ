#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *
#show: codly-init.with()
#codly(languages: codly-languages)

#set heading(numbering: "1. 1. 1.")

#set text(
  size: 9pt
)

#set page(columns: 1,
margin: (
  top: 3cm,
  bottom: 2cm,
  x: 0.8cm,
))

#show "Rta": name => box[
  #highlight(fill: green)[_Respuesta_]
]

#set text(size: 11pt)

= Resolver el problema de clasificación de lesiones dermatológicas pero esta vez utilizando CNNs 
\
Para ello, se replantea la implementación previamente desarrollada con un MLP, adaptándola a una arquitectura CNN lo más simple posible. El modelo está inspirado en AlexNet y se construye desde cero utilizando PyTorch, sin aplicar técnicas de transfer learning, data augmentation ni métodos avanzados de regularización.

La arquitectura empleada incluye:

 - capas convolucionales
 - funciones de activación ReLU
 - operaciones de MaxPooling
 - capas fully connected.

Además, las imágenes son redimensionadas a 128x128 píxeles con el objetivo de reducir el costo computacional del entrenamiento.

El modelo se entrena utilizando el optimizador Adam y la función de pérdida CrossEntropyLoss, adecuada para problemas de clasificación multiclase.

A partir de esta configuración básica, se obtienen los siguientes resultados, los cuales servirán como baseline inicial para futuras mejoras y comparaciones.

el archivo '01_baseline_cnn.ipynb' nos proporciona los siguentes resultados:

```
                              precision    recall  f1-score   support

         Actinic keratosis       0.41      0.65      0.50        20
         Atopic Dermatitis       0.64      0.33      0.44        21
          Benign keratosis       0.90      0.95      0.93        20
            Dermatofibroma       0.47      0.40      0.43        20
         Melanocytic nevus       0.62      0.75      0.68        20
                  Melanoma       0.47      0.45      0.46        20
   Squamous cell carcinoma       0.25      0.25      0.25        20
Tinea Ringworm Candidiasis       0.55      0.55      0.55        20
           Vascular lesion       0.94      0.80      0.86        20

                  accuracy                           0.57       181
                 macro avg       0.58      0.57      0.57       181
              weighted avg       0.58      0.57      0.57       181

```

#figure(
  image("images/baselineMatrix.png", width: 70%),
  caption: "Matriz de Confusion "
) <fig:ConfBase>

#figure(
  image("images/graphBase.png", width: 70%),
  caption: "Iteraciones de baseline_cnn"
) <fig:ConfBase>

#pagebreak()
= Agregando augmentation
\

Modificando transform por:
```python
train_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(
        brightness=0.2
    ),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])
```

Obtenemos los siguientes resultados

```
                              precision    recall  f1-score   support

         Actinic keratosis       0.50      0.65      0.57        20
         Atopic Dermatitis       0.48      0.76      0.59        21
          Benign keratosis       0.89      0.85      0.87        20
            Dermatofibroma       0.61      0.55      0.58        20
         Melanocytic nevus       0.67      0.90      0.77        20
                  Melanoma       0.43      0.45      0.44        20
   Squamous cell carcinoma       0.27      0.15      0.19        20
Tinea Ringworm Candidiasis       0.38      0.15      0.21        20
           Vascular lesion       0.94      0.85      0.89        20

                  accuracy                           0.59       181
                 macro avg       0.58      0.59      0.57       181
              weighted avg       0.57      0.59      0.57       181
```

#figure(
  image("images/matriz2.png", width: 70%),
  caption: "Matriz de Confusion"
) <fig:ConfBase>

#figure(
  image("images/iteraciones2.png", width: 70%),
  caption: "Iteraciones comparando base con augmented"
) <fig:ConfBase>

#pagebreak()
= Agregando regularizacion
\
Agergando normalizaciones y un dropout de 0.2 en el arhivo '03_regularization.ipynb', obtenemos los siguientes resultados:


```
                              precision  recall  f1-score   support

         Actinic keratosis       0.36      0.70      0.47        20
         Atopic Dermatitis       0.53      0.81      0.64        21
          Benign keratosis       0.95      0.95      0.95        20
            Dermatofibroma       0.33      0.30      0.32        20
         Melanocytic nevus       0.75      0.75      0.75        20
                  Melanoma       0.50      0.40      0.44        20
   Squamous cell carcinoma       0.40      0.20      0.27        20
Tinea Ringworm Candidiasis       0.38      0.15      0.21        20
           Vascular lesion       0.89      0.80      0.84        20

                  accuracy                           0.56       181
                 macro avg       0.57      0.56      0.54       181
              weighted avg       0.57      0.56      0.54       181
```

#figure(
  image("images/matriz3.png", width: 70%),
  caption: "Matriz de Confusion"
) <fig:ConfBase>

#figure(
  image("images/graph3.png", width: 70%),
  caption: "Iteraciones comparando regularizacion con"
) <fig:ConfBase>





#pagebreak()
= Busqueda de Hiperparametros
\
Iterando los parametros con 10 iteraciones, obtenemos la sieguiente tabla

```
      lr    batch_size  dropout  optimizer accuracy
24  0.0001          16      0.3      Adam  57.458564
30  0.0001          32      0.5      Adam  56.906077
28  0.0001          32      0.3      Adam  55.801105
17  0.0010          32      0.3       SGD  55.801105
23  0.0010          64      0.5       SGD  54.696133
13  0.0010          16      0.3       SGD  54.696133
34  0.0001          64      0.5      Adam  54.143646
21  0.0010          64      0.3       SGD  54.143646
32  0.0001          64      0.3      Adam  53.591160
12  0.0010          16      0.3      Adam  52.486188
19  0.0010          32      0.5       SGD  51.933702
26  0.0001          16      0.5      Adam  51.381215
16  0.0010          32      0.3      Adam  50.828729
15  0.0010          16      0.5       SGD  49.723757
22  0.0010          64      0.5      Adam  49.171271
25  0.0001          16      0.3       SGD  48.618785
29  0.0001          32      0.3       SGD  47.513812
9   0.0100          64      0.3       SGD  47.513812
27  0.0001          16      0.5       SGD  45.856354
5   0.0100          32      0.3       SGD  44.751381
14  0.0010          16      0.5      Adam  44.751381
31  0.0001          32      0.5       SGD  43.646409
18  0.0010          32      0.5      Adam  43.646409
11  0.0100          64      0.5       SGD  42.541436
20  0.0010          64      0.3      Adam  42.541436
33  0.0001          64      0.3       SGD  41.988950
35  0.0001          64      0.5       SGD  37.569061
7   0.0100          32      0.5       SGD  32.596685
1   0.0100          16      0.3       SGD  28.176796
3   0.0100          16      0.5       SGD  24.861878
8   0.0100          64      0.3      Adam  16.022099
4   0.0100          32      0.3      Adam  12.707182
10  0.0100          64      0.5      Adam  12.707182
0   0.0100          16      0.3      Adam  11.602210
2   0.0100          16      0.5      Adam  11.602210
6   0.0100          32      0.5      Adam  11.049724
```

Observamos que la mejor configuracion entonces, es con un  'lr': 0.0001, 'batch_size': 16, 'dropout': 0.3, 'optimizer': 'Adam' obteniendo un 'accuracy' de 57.4585635359116