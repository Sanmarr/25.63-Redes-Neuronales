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

= Preguntas sobre el ejemplo de clasificación de imágenes con PyTorch y MLP

== Dataset y Preprocesamiento

=== ¿Por qué es necesario redimensionar las imágenes a un tamaño fijo para una MLP?

Rta: 

Las MLP (Multi-Layer Perceptrons) requieren que todas las entradas tengan la misma dimensión. Redimensionar las imágenes a un tamaño fijo asegura que cada imagen tenga el mismo número de píxeles, lo que permite que la MLP procese los datos de manera consistente. Si las imágenes tuvieran tamaños variables, la MLP no podría manejar la entrada correctamente.


=== ¿Qué ventajas ofrece Albumentations frente a otras librerías de transformación como `torchvision.transforms`?

Rta: 

Albumentations es una librería especializada en augmentación de datos que ofrece varias ventajas sobre torchvision.transforms al ser mas moderna. Esta esta mas optimizada, es mas rapida y reduce el cuello de botella en el preprocesamiento.

=== ¿Qué hace `A.Normalize()`? ¿Por qué es importante antes de entrenar una red?

Rta: 

`A.Normalize()` normaliza los valores de píxeles de la imagen restando la media (mean) y dividiendo por la desviación estándar (std) de los canales. Permite utilizar la distribucion normal estandar modificando los valores originales de las entradas. Es una regla de oro antes de entrenar una red, ya que ayuda a estabilizar y acelerar el proceso de entrenamiento al asegurar que los datos estén en un rango similar, lo que facilita la convergencia del modelo.

=== ¿Por qué convertimos las imágenes a `ToTensorV2()` al final de la pipeline?

Rta: 

Porque PyTorch requiere tensores, no arrays. Se implementa al final de la pipeline para asegurarnos de que todas las transformaciones anteriores se apliquen a los datos en formato de imagen (arrays). Luego, para el entrenamiento, se requieren en formato tensor.




== 2. Arquitectura del Modelo

=== ¿Por qué usamos una red MLP en lugar de una CNN aquí? ¿Qué limitaciones tiene?

Rta: 

En este caso, sería preferible utilizar una red CNN para clasificación de imagenes, ya que las CNN están diseñadas para reconocer caracteristicas espaciales y patrones en las imagenes. Sin embargo, se utilizo MLP debido a la baja complejidad computacional, al ser un pequeño dataset y por motivos educativos, para compararla despues con una CNN.
Una de las limitaciones mas importantes al utilizar redes MLP, es que pierde la referencia espacial de las imagenes al tener que convertirlas en un vector unidimensional. CNN puede detectar bordes, esquinas, texturas en la vecindad mientras que MLP ve cada píxel de forma independiente.


=== ¿Qué hace la capa `Flatten()` al principio de la red?

Rta:



=== ¿Qué función de activación se usó? ¿Por qué no usamos `Sigmoid` o `Tanh`?

Rta:

=== ¿Qué parámetro del modelo deberíamos cambiar si aumentamos el tamaño de entrada de la imagen?

Rta:






== 3. Entrenamiento y Optimización

=== ¿Qué hace `optimizer.zero_grad()`?

Rta:

=== ¿Por qué usamos `CrossEntropyLoss()` en este caso?

Rta:

=== ¿Cómo afecta la elección del tamaño de batch (`batch_size`) al entrenamiento?

Rta:

=== ¿Qué pasaría si no usamos `model.eval()` durante la validación?

Rta:





== 4. Validación y Evaluación

=== ¿Qué significa una accuracy del 70% en validación pero 90% en entrenamiento?

Rta:

===  ¿Qué otras métricas podrían ser más relevantes que accuracy en un problema real?

Rta:

===  ¿Qué información útil nos da una matriz de confusión que no nos da la accuracy?

Rta:

===  En el reporte de clasificación, ¿qué representan `precision`, `recall` y `f1-score`?

Rta:





== 5. TensorBoard y Logging

=== ¿Qué ventajas tiene usar TensorBoard durante el entrenamiento?

Rta:

=== ¿Qué diferencias hay entre loguear `add_scalar`, `add_image` y `add_text`?

Rta:

=== ¿Por qué es útil guardar visualmente las imágenes de validación en TensorBoard?

Rta:

=== ¿Cómo se puede comparar el desempeño de distintos experimentos en TensorBoard?

Rta:


== 6. Generalización y Transferencia

=== ¿Qué cambios habría que hacer si quisiéramos aplicar este mismo modelo a un dataset con 100 clases?

Rta:

===  ¿Por qué una CNN suele ser más adecuada que una MLP para clasificación de imágenes?

Rta:

===  ¿Qué problema podríamos tener si entrenamos este modelo con muy pocas imágenes por clase?

Rta:

===  ¿Cómo podríamos adaptar este pipeline para imágenes en escala de grises?

Rta:


= Regularización

== Preguntas teóricas

=== ¿Qué es la regularización en el contexto del entrenamiento de redes neuronales?

Rta:


===  ¿Cuál es la diferencia entre `Dropout` y regularización `L2` (weight decay)?

Rta:

===  ¿Qué es `BatchNorm` y cómo ayuda a estabilizar el entrenamiento?

Rta:

===  ¿Cómo se relaciona `BatchNorm` con la velocidad de convergencia?

===  ¿Puede `BatchNorm` actuar como regularizador? ¿Por qué?

Rta:

===  ¿Qué efectos visuales podrías observar en TensorBoard si hay overfitting?

Rta:

===  ¿Cómo ayuda la regularización a mejorar la generalización del modelo?

Rta:




== Actividades de modificación:

=== Agregar Dropout en la arquitectura MLP
- Insertar capas `nn.Dropout(p=0.5)` entre las capas lineales y activaciones.
- Comparar los resultados con y sin `Dropout`.

=== Agregar Batch Normalization
Insertar `nn.BatchNorm1d(...)` después de cada capa `Linear` y antes de la activación:
     ```python
     self.net = nn.Sequential(
         nn.Flatten(),
         nn.Linear(in_features, 512),
         nn.BatchNorm1d(512),
         nn.ReLU(),
         nn.Dropout(0.5),
         nn.Linear(512, 256),
         nn.BatchNorm1d(256),
         nn.ReLU(),
         nn.Dropout(0.5),
         nn.Linear(256, num_classes)
     )
     ```

=== Aplicar Weight Decay (L2)
Modificar el optimizador:
 ```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001,
weight_decay=1e-4)
```
=== Reducir overfitting con data augmentation
Agregar transformaciones en Albumentations como `HorizontalFlip`, `BrightnessContrast`, `ShiftScaleRotate`.

=== Early Stopping (opcional)
Implementar un criterio para detener el entrenamiento si la validación no mejora después de N épocas.




== Preguntas prácticas:

=== ¿Qué efecto tuvo `BatchNorm` en la estabilidad y velocidad del entrenamiento?

=== ¿Cambió la performance de validación al combinar `BatchNorm` con `Dropout`?

=== ¿Qué combinación de regularizadores dio mejores resultados en tus pruebas?

=== ¿Notaste cambios en la loss de entrenamiento al usar `BatchNorm`?




= Inicialización de Parámetros

== Preguntas teóricas:

===  ¿Por qué es importante la inicialización de los pesos en una red neuronal?

===  ¿Qué podría ocurrir si todos los pesos se inicializan con el mismo valor?

===  ¿Cuál es la diferencia entre las inicializaciones de Xavier (Glorot) y He?

===  ¿Por qué en una red con ReLU suele usarse la inicialización de He?

===  ¿Qué capas de una red requieren inicialización explícita y cuáles no?


== Actividades de modificación:

=== Agregar inicialización manual en el modelo
En la clase `MLP`, agregar un método `init_weights` que inicialice cada capa:
  ```python
  def init_weights(self):
      for m in self.modules():
          if isinstance(m, nn.Linear):
              nn.init.kaiming_normal_(m.weight)
              nn.init.zeros_(m.bias)
  ```


=== Probar distintas estrategias de inicialización
```python
   - Xavier (`nn.init.xavier_uniform_`)
   - He (`nn.init.kaiming_normal_`)
   - Aleatoria uniforme (`nn.init.uniform_`)
   - Comparar la estabilidad y velocidad del entrenamiento.
```

=== Visualizar pesos en TensorBoard:
Agregar esta línea en la primera época para observar los histogramas:
 ```python
 for name, param in model.named_parameters():
     writer.add_histogram(name, param, epoch)
 ```

==  Preguntas prácticas:

=== ¿Qué diferencias notaste en la convergencia del modelo según la inicialización?

=== ¿Alguna inicialización provocó inestabilidad (pérdida muy alta o NaNs)?

=== ¿Qué impacto tiene la inicialización sobre las métricas de validación?

=== ¿Por qué `bias` se suele inicializar en cero?


