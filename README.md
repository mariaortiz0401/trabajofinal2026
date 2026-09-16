# Caracterización del proceso de aprendizaje de un curso de programación de computadores con Python mediante analítica del aprendizaje

Este repositorio contiene el código fuente, notebooks de procesamiento, análisis exploratorio, modelado predictivo y scripts de experimentación pertenecientes al Trabajo Final de Maestría titulado:

> **«Caracterización del proceso de aprendizaje de un curso de programación de computadores con Python mediante analítica del aprendizaje»**  
> **Autora:** María Alejandra Ortiz Mora  
> **Programa:** Maestría en Ingeniería de Sistemas y Computación  
> **Institución:** Universidad Nacional de Colombia — Facultad de Ingeniería  
> **Director:** Felipe Restrepo Calle, Ph.D.  
> **Año:** 2026  

---

## Descripción del Proyecto

El proyecto aplica técnicas de **Learning Analytics (LA)**, **Procesamiento del Lenguaje Natural (PLN)** y **Aprendizaje Automático (Machine Learning)** sobre un conjunto de datos recopilado a lo largo de 5 años (2021–2025) en el curso masivo y abierto *«Introducción a la Programación con Python»* de la Universidad Nacional de Colombia.

A partir de más de 176.000 trazas de interacción en la plataforma de evaluación automática **UNCode**, encuestas sociodemográficas de caracterización y cuestionarios de autoevaluación post-curso, el estudio abarca:
- **Estandarización y unificación** de registros provenientes de dos entornos educativos (*EduNext* y *OpenEdX*).
- **Análisis Exploratorio de Datos (EDA)** y caracterización de perfiles estudiantiles (*Aprobados*, *Reprobados* y *Desertores*).
- **Entrenamiento de modelos de clasificación** para la predicción del desempeño académico (Día 0 vs. modelos de tiempo/evento tarea a tarea).
- **Detección zero-shot de código generado por Inteligencia Artificial** (*DetectCodeGPT*) y evaluación estadística no paramétrica (*Kruskal-Wallis* y *Dunn*) sobre la evolución del uso de LLMs (2022–2025).

---

## Estructura del Repositorio

El flujo de trabajo se encuentra modularizado en notebooks secuenciales y scripts complementarios para garantizar su reproducibilidad:

| Archivo / Notebook | Descripción |
| :--- | :--- |
| `1_import_data.ipynb` | Importación y consolidación inicial de fuentes CSV de EduNext y OpenEdX. |
| `1_2_UNCODE_recovery_caracterization.ipynb` | Recuperación y mapeo de identificadores de usuario UNCode en encuestas de caracterización. |
| `1_3_UNCODE_recovery_evaluation.ipynb` | Recuperación y mapeo de identificadores de usuario UNCode en encuestas de autoevaluación. |
| `2_standardize_data.ipynb` | Limpieza, estandarización de columnas y unificación de datasets. |
| `3_analyze_data.ipynb` | Análisis Exploratorio de Datos (EDA), segmentación por grupos de desempeño y métricas de retención. |
| `4_Models.ipynb` | Construcción de perfiles (*Feature Engineering*), entrenamiento de Random Forest y SVM (Día 0 y modelos tiempo/evento). |
| `5_IA_detection_fork.ipynb` | Análisis de resultados del detector de IA, pruebas de Kruskal-Wallis y post-hoc de Dunn por cohortes anuales. |
| `run-pipeline-opt-batch-final-0719.py` | Script de optimización por lotes para ejecución masiva en GPU del modelo de detección de IA en Python. |

---

## Requisitos e Instalación

### Entorno y Dependencias Principales
- **Lenguaje:** Python 3.9+
- **Librerías de Procesamiento y Machine Learning:** `pandas`, `numpy`, `scikit-learn`, `scipy`, `scikit-posthocs`
- **Procesamiento de Texto y PLN:** `nltk`, `thefuzz`, `pysentimiento`, `transformers`, `torch`
- **Visualización:** `matplotlib`, `seaborn`
