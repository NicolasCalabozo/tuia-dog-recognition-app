from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingRecord(BaseModel):
    """Registro de la base vectorial (estructura sugerida por el TP)."""

    id_imagen: str
    embedding: list[float]
    path: str
    breed: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Neighbor(BaseModel):
    """Un vecino recuperado por la busqueda por similitud."""

    path: str
    breed: str
    score: float
    url: Optional[str] = Field(
        default=None,
        description="URL relativa al API para descargar la imagen (/files/data/...).",
    )


class SearchRequest(BaseModel):
    source_path: str
    model: Optional[str] = Field(
        default=None,
        description="baseline | resnet18_finetuned | cnn_custom (default: EMBEDDING_MODEL).",
    )
    top_k: Optional[int] = Field(default=None, description="Cantidad de vecinos (default: TOP_K).")


class SearchResult(BaseModel):
    """Resultado de la Etapa 1: imagen consultada, top K similares y raza predicha."""

    type: Literal["search"] = "search"
    source_path: str
    model: str
    predicted_breed: str
    score: float
    neighbors: list[Neighbor]


class ClassifyRequest(BaseModel):
    source_path: str
    model: Optional[str] = Field(
        default=None,
        description="resnet18_finetuned | cnn_custom (default: resnet18_finetuned).",
    )


class ClassifyResult(BaseModel):
    """Resultado de la Etapa 2: raza predicha por el clasificador entrenado."""

    type: Literal["classify"] = "classify"
    source_path: str
    model: str
    breed: str
    score: float


class DetectRequest(BaseModel):
    source_path: str


class DogDetection(BaseModel):
    """Una deteccion del pipeline (Etapa 3)."""

    bbox: list[int]
    det_score: float
    breed: str
    breed_score: float


class DetectResult(BaseModel):
    """Resultado de la Etapa 3: bounding boxes, razas y scores de confianza."""

    type: Literal["detect"] = "detect"
    source_path: str
    detections: list[DogDetection]
    detected_breeds: list[str]


class AsyncTaskCreated(BaseModel):
    status: Literal["accepted"] = "accepted"
    job_id: str


class UploadResponse(BaseModel):
    """Respuesta tras subir un archivo al servidor (rutas usadas por /search y /detect)."""

    path: str
    download_url: str


class StatusResponse(BaseModel):
    status: Literal["done", "inProgress", "failed"]
    link: str
    reason: Optional[str] = None
    artifact_url: Optional[str] = Field(
        default=None,
        description="URL relativa al API del artefacto principal (.json del resultado).",
    )
    source_image_url: Optional[str] = Field(
        default=None,
        description="URL relativa de la imagen origen (consulta o imagen procesada).",
    )


class ModelsResponse(BaseModel):
    models: list[str]
    selected: str

class BloqueResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BloqueResidual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CNNCustom(nn.Module):
    def __init__(self, num_clases=70):
        super(CNNCustom, self).__init__()

        self.conv_inicial = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn_inicial = nn.BatchNorm2d(64)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.capa1 = self._crear_capa(in_channels=64, out_channels=64, stride=1, num_bloques=2)
        self.capa2 = self._crear_capa(in_channels=64, out_channels=128, stride=2, num_bloques=2)
        self.capa3 = self._crear_capa(in_channels=128, out_channels=256, stride=2, num_bloques=2)
        self.capa4 = self._crear_capa(in_channels=256, out_channels=512, stride=2, num_bloques=2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(512, num_clases)

    def _crear_capa(self, in_channels, out_channels, stride, num_bloques):
        capas = []
        capas.append(BloqueResidual(in_channels, out_channels, stride))
        for _ in range(1, num_bloques):
            capas.append(BloqueResidual(out_channels, out_channels, stride=1))
        return nn.Sequential(*capas)

    def forward(self, x):

        x = F.relu(self.bn_inicial(self.conv_inicial(x)))
        x = self.maxpool(x)
        x = self.capa1(x)
        x = self.capa2(x)
        x = self.capa3(x)
        x = self.capa4(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)

        x = self.dropout(x)
        x = self.fc(x)

        return x