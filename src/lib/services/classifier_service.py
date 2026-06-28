from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from ..schemas import CNNCustom
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import onnxruntime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import cv2
from PIL import Image
logger = logging.getLogger(__name__)


class ClassifierService:
    """Etapa 2: entrenamiento y comparacion de modelos de clasificacion.

    Funciones a implementar por el estudiante:
      - train_classifier()
      - evaluate_classifier()
      - extract_custom_embedding(image)

    La carga de checkpoints (.pth / .onnx) y la seleccion del modelo activo
    ya estan provistas.
    """

    def __init__(
        self,
        checkpoints: dict[str, Path],
        image_size: int,
        dataset_path: Path,
        output_path: Path,
        active_model: str = "resnet18_finetuned",
    ) -> None:
        # checkpoints: nombre logico -> ruta del archivo (ej. resnet18_finetuned -> models/resnet18_finetuned.pth)
        self.checkpoints = checkpoints
        self.image_size = image_size
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.active_model_name = active_model
        self._loaded: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Infraestructura provista
    # ------------------------------------------------------------------

    def set_active_model(self, name: str) -> None:
        """Define que checkpoint usan extract_custom_embedding y la clasificacion.

        Valores esperados: resnet18_finetuned | cnn_custom.
        """
        if name not in self.checkpoints:
            raise ValueError(f"Unknown model '{name}'. Expected one of: {sorted(self.checkpoints)}")
        self.active_model_name = name

    @property
    def active_checkpoint(self) -> Path:
        return self.checkpoints[self.active_model_name]

    def load_model(self, name: str | None = None) -> Any:
        """Carga (con cache) el checkpoint del modelo indicado o del activo.

        Soporta modelos PyTorch (.pth) y exportados a ONNX (.onnx).
        """
        key = name or self.active_model_name
        if key in self._loaded:
            return self._loaded[key]
        
        path = self.checkpoints[key]
        if not path.exists():
            raise ValueError(
                f"Checkpoint not found: {path}. Entrena el modelo (Etapa 2) y guardalo en esa ruta."
            )
            
        suf = path.suffix.lower()
        if suf == ".pth":
            # Cargamos el diccionario completo (checkpoint)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            
            # Averiguamos el número de clases leyendo las subcarpetas del dataset de entrenamiento
            train_dir = self.dataset_path / 'train'
            num_clases = len([d for d in train_dir.iterdir() if d.is_dir()])
            
            # Instanciamos la arquitectura correcta según el 'key' del modelo activo
            if key == "resnet18_finetuned":
                model = models.resnet18() # No necesitamos weights=DEFAULT porque cargaremos los nuestros
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, num_clases)
                
            elif key == "cnn_custom":
                model = CNNCustom(num_clases=num_clases)
                
            else:
                raise ValueError(f"Modelo no soportado para carga: {key}")
            
            # Inyectamos los pesos guardados en la arquitectura instanciada
            model.load_state_dict(checkpoint['modelo_estado'])

        elif suf == ".onnx":
            model = onnxruntime.InferenceSession(str(path))
        else:
            raise ValueError(f"Unsupported model format (expected .pth or .onnx): {path}")
            
        self._loaded[key] = model
        return model

    # ------------------------------------------------------------------
    # Etapa 2: funciones a implementar
    # ------------------------------------------------------------------

    def train_classifier(self, num_epocas: int = 100) -> dict[str, list[float]]:
        """
        Entrena el clasificador de razas sobre el dataset (self.dataset_path).
        
        Hiperparámetros configurados:
        - Learning Rate: 0.001 (Adam optimizer)
        - Batch Size: Definido por variable de entorno BATCH_SIZE
        - Epochs: {num_epocas}
        - Optimizador: Adam
        - Scheduler: ReduceLROnPlateau (patience=5, mode='min')
        - Early Stopping: Activado (paciencia=15)
        
        El modelo resultante se guarda en: {self.active_checkpoint}
        """
        # Establecemos el preprocesamiento de imágenes para adaptar a Resnet18 y CNN custom
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # Configuramos el dispositivo (GPU o CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #Logging de inicio de entrenamiento
        print(f"Iniciando entrenamiento en: {device}")
        print(f"Modelo activo: {self.active_model_name}")

        #Para evaluar el train vs. valid loss hay que mantener un historial
        historial = {
            'train_loss': [],
            'valid_loss': [],
            'train_acc': [],
            'valid_acc': []
        }
        #Seteo de path para dataset de train por Pathlib
        train_dir = self.dataset_path / 'train'
        valid_dir = self.dataset_path / 'valid'

        #Preparación de los dataloaders
        batch_size = int(os.getenv("BATCH_SIZE", "32"))

        train_dataset = datasets.ImageFolder(root=train_dir, transform=preprocess)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        valid_dataset = datasets.ImageFolder(root=valid_dir, transform=preprocess)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        num_clases = len(train_dataset.classes)
        print(f"\nClases detectadas : {num_clases}")
        print(f"Usando un batch size de: {batch_size}\n")
        #Establecemos fases para el entrenamiento y validación
        fases = {'entrenamiento': train_loader, 'validacion': valid_loader}
        

        # Selección e instanciación del modelo
        if self.active_model_name == "resnet18_finetuned":
            #Cargamos el modelo preentrenado ResNet18 y congelamos todas las capas excepto la última
            modelo = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

            for param in modelo.parameters():
                param.requires_grad = False
                
            for param in modelo.layer4.parameters():
                param.requires_grad = True

            num_ftrs = modelo.fc.in_features
            #Reemplazamos la capa final para que coincida con el número de clases de nuestro dataset
            modelo.fc = nn.Linear(num_ftrs, num_clases)
        elif self.active_model_name == "cnn_custom":
            #Instanciamos nuestro modelo CNN customizado
            modelo = CNNCustom(num_clases=num_clases) 
        else:
            raise ValueError(f"Modelo no soportado para entrenamiento: {self.active_model_name}")
        
        print(f"Modelo Instanciado: {self.active_model_name}")
        modelo = modelo.to(device)

        #Configuración de la función de pérdida, optimizador y scheduler
        criterio = nn.CrossEntropyLoss()
        parametros_a_entrenar = filter(lambda p: p.requires_grad, modelo.parameters())
        optimizador = optim.Adam(parametros_a_entrenar, lr=0.001)
        programador = optim.lr_scheduler.ReduceLROnPlateau(optimizador, patience=5)

        #Lógica de Entrenamiento
        inicio = time.time()
        mejor_perdida = float('inf')
        paciencia_early_stopping = 15
        epocas_sin_mejora = 0

        for epoca in range(num_epocas):
            print(f'\nÉpoca {epoca + 1}/{num_epocas}')
            print('-' * 10)

            for fase in ['entrenamiento', 'validacion']:
                if fase == 'entrenamiento':
                    modelo.train()
                else:
                    modelo.eval()

                perdida_acumulada = 0.0
                aciertos_acumulados = 0
                total_muestras = 0

                for entradas, etiquetas in fases[fase]:
                    entradas = entradas.to(device)
                    etiquetas = etiquetas.to(device)

                    optimizador.zero_grad()

                    with torch.set_grad_enabled(fase == 'entrenamiento'):
                        salidas = modelo(entradas)
                        _, predicciones = torch.max(salidas, 1)
                        perdida = criterio(salidas, etiquetas)

                        if fase == 'entrenamiento':
                            perdida.backward()
                            optimizador.step()
                    
                    perdida_acumulada += perdida.item() * entradas.size(0)
                    aciertos_acumulados += torch.sum(predicciones == etiquetas.data).item()
                    total_muestras += entradas.size(0)

                perdida_epoca = perdida_acumulada / total_muestras
                precision_epoca = aciertos_acumulados / total_muestras

                print(f'{fase.capitalize()} - Pérdida: {perdida_epoca:.4f} Precisión: {precision_epoca:.4f}')
                
                #Guardamos el historial de pérdidas y precisión para cada fase
                if fase == 'entrenamiento':
                    historial['train_loss'].append(perdida_epoca)
                    historial['train_acc'].append(precision_epoca)
                else:
                    historial['valid_loss'].append(perdida_epoca)
                    historial['valid_acc'].append(precision_epoca)
                
                if fase == 'validacion':
                    programador.step(perdida_epoca)

                    # Guardar el mejor modelo usando self.active_checkpoint
                    if perdida_epoca < mejor_perdida:
                        mejor_perdida = perdida_epoca
                        epocas_sin_mejora = 0

                        self.active_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                        torch.save({
                            'epoca': epoca,
                            'modelo_estado': modelo.state_dict(),
                            'optimizador_estado': optimizador.state_dict(),
                            'perdida': mejor_perdida,
                        }, self.active_checkpoint)
                        print(f"-> Mejora - Checkpoint guardado")
                    else:
                        epocas_sin_mejora += 1
                        print(f"-> Sin mejoras en validación. Paciencia Early Stopping: {epocas_sin_mejora}/{paciencia_early_stopping}")

            if epocas_sin_mejora >= paciencia_early_stopping:
                print(f'\n[!] Early Stopping activado. El modelo dejó de aprender durante {paciencia_early_stopping} épocas consecutivas.')
                break

        tiempo_transcurrido = time.time() - inicio
        print(f'\nEntrenamiento completado en {tiempo_transcurrido // 60:.0f}m {tiempo_transcurrido % 60:.0f}s')
        print(f'Mejor Pérdida en Validación: {mejor_perdida:.4f}')

        archivo_historial = self.output_path / f"{self.active_model_name}_history.json"
        self.output_path.mkdir(parents=True, exist_ok=True) 
        
        with open(archivo_historial, 'w') as f:
            json.dump(historial, f, indent=4)
            
        print(f"Historial guardado exitosamente en: {archivo_historial}")

        return historial

    def evaluate_classifier(self) -> dict[str, float]:
        """
        Evalua el modelo activo sobre el conjunto de prueba.

        Retorna accuracy, precision, recall (sensibilidad),
        specificity (especificidad) y F1-Score. Incluye la matríz de confusión para análisis detallado.
        
        """
        preprocesamiento = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        test_dir = self.dataset_path / 'test'
        if not test_dir.exists():
            raise FileNotFoundError(f"No se encontró el directorio de prueba: {test_dir}")
        
        test_dataset = datasets.ImageFolder(root=test_dir, transform=preprocesamiento)
        batch_size = int(os.getenv("BATCH_SIZE", "32"))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        modelo = self.load_model()
        modelo = modelo.to(device)
        modelo.eval()

        print(f"Evaluando modelo '{self.active_model_name}' sobre {len(test_dataset)} imágenes")
        y_true = []
        y_pred = []
        y_probs = []

        with torch.no_grad():
                for entradas, etiquetas in test_loader:
                    entradas = entradas.to(device)
                    salidas = modelo(entradas)
                    _, predicciones = torch.max(salidas, 1)
                    y_true.extend(etiquetas.cpu().numpy())
                    y_pred.extend(predicciones.cpu().numpy())
                    probabilidades = torch.softmax(salidas, dim=1)
                    y_probs.extend(probabilidades.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        rec = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        cm = confusion_matrix(y_true, y_pred)

        # Para cada clase extraemos sus valores a partir de la matriz de confusión
        fp = cm.sum(axis=0) - np.diagonal(cm)  # Falsos Positivos
        fn = cm.sum(axis=1) - np.diagonal(cm)  # Falsos Negativos
        tp = np.diagonal(cm)                   # Verdaderos Positivos
        tn = cm.sum() - (fp + fn + tp)         # Verdaderos Negativos
        
        # Specificity = TN / (TN + FP).
        specificity_per_class = np.divide(tn, (tn + fp), out=np.zeros_like(tn, dtype=float), where=(tn + fp) != 0)
        spec = float(np.mean(specificity_per_class))

        resultados = {
            "accuracy": round(acc, 2),
            "precision": round(prec, 2),
            "recall": round(rec, 2),
            "specificity": round(spec, 2),
            "f1": round(f1, 2),
            "classes": test_dataset.classes,
            "confusion_matrix": cm.tolist(),
            "y_true": np.array(y_true).tolist(),
            "y_probs": np.array(y_probs).tolist()
        }
        
        archivo_metricas = self.output_path / f"{self.active_model_name}_metrics.json"
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        with open(archivo_metricas, 'w') as f:
            json.dump(resultados, f, indent=4)
            
        print(f"Métricas guardadas exitosamente en: {archivo_metricas}")

        return resultados
    
def extract_custom_embedding(self, image: np.ndarray) -> list[float]:
        """
        Genera el embedding a partir del modelo activo - Implementada pero no utilizada
        Reemplaza la última capa por una identidad (dummy) para poder extraer el embedding.

        TO-DO: 
        - Agregar la instancia de modelo a __init__ para no cargar el modelo en caso de extracción masiva
        - Extraer embeddings de Test
        - Contrastarlos contra la BBDD vectorial
        - Observar agrupación de resultados utilizando PCA y T-SNE o algoritmos de clusterización
        - Al día de hoy es incompatible con la base de datos vectorial de Etapa 1 por la diferencia
            en dimensionalidad de los embeddings 1280 de EfficientNetB0 vs. 512 de Resnet18 y CNN Custom.
        """

        imagen_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imagen_pil = Image.fromarray(imagen_rgb)

        preprocesamiento = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        input_tensor = preprocesamiento(imagen_pil).unsqueeze(0) #type: ignore

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        modelo = self.load_model()
        modelo = modelo.to(device)
        modelo.eval()
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            if self.active_model_name == "resnet18_finetuned":
                
                capa_original = modelo.fc 
                modelo.fc = nn.Identity() 
                embedding_tensor = modelo(input_tensor)
                modelo.fc = capa_original 
                
            elif self.active_model_name == "cnn_custom":

                capa_original = modelo.fc 
                modelo.fc = nn.Identity()
                embedding_tensor = modelo(input_tensor)
                modelo.fc = capa_original
            else:
                raise ValueError("Modelo no soportado.")

        vector = embedding_tensor.cpu().numpy().flatten().tolist()

        return vector