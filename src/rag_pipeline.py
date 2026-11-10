"""
Pipeline RAG para el Asistente Documental de Metro Bilbao.
Combina recuperación vectorial (Embeddings BGE + Cosine Similarity) con
generación condicionada por contexto (LLM Text2Text / Causal).
Configurado exclusivamente para operar con modelos de la familia BGE (ej: BAAI/bge-m3).
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

MODELO_EMBEDDING_POR_DEFECTO = "BAAI/bge-m3"


class MetroBilbaoRAG:
    def __init__(
        self,
        embedding_model_name: str = MODELO_EMBEDDING_POR_DEFECTO,
        generation_model_name: str = "google/flan-t5-small",
        device: Optional[int] = None,
        umbral_similitud_defecto: float = 0.45
    ):
        """
        Inicializa el pipeline RAG.
        :param embedding_model_name: Nombre o ruta del modelo de embeddings (debe ser de la familia BGE).
        :param generation_model_name: Nombre o ruta del modelo generativo.
        :param device: Dispositivo para inferencia (0 para GPU cuda:0, -1 para CPU).
        :param umbral_similitud_defecto: Puntuación mínima de similitud requerida (0.45 calibrado para BGE).
        """
        if "bge" not in embedding_model_name.lower():
            raise ValueError(
                f"Modelo no permitido: '{embedding_model_name}'. "
                "Este proyecto está configurado para operar exclusivamente con la familia de modelos BGE (ej: 'BAAI/bge-m3')."
            )

        self.embedding_model_name = embedding_model_name
        self.generation_model_name = generation_model_name
        self.umbral_similitud = umbral_similitud_defecto

        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = device

        print(f"[RAG] Cargando modelo de embeddings: {embedding_model_name}...")
        self.modelo_embeddings = SentenceTransformer(embedding_model_name)

        print(f"[RAG] Cargando generador LLM: {generation_model_name} en dispositivo {self.device}...")
        self.generador = pipeline(
            task="text2text-generation",
            model=generation_model_name,
            device=self.device
        )

        self.documentos: List[Dict[str, Any]] = []
        self.embeddings_documentos: Optional[np.ndarray] = None
        print("[RAG] Sistema inicializado correctamente.")

    def indexar_documentos(self, documentos: List[Dict[str, Any]]):
        """
        Genera los embeddings de los documentos y los almacena en memoria.
        """
        self.documentos = documentos
        textos = [doc.get("texto_completo", doc["texto"]) for doc in self.documentos]

        print(f"[RAG] Indexando {len(textos)} fragmentos documentales...")
        self.embeddings_documentos = self.modelo_embeddings.encode(
            textos,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        print(f"[RAG] Indexación completada. Matriz de embeddings: {self.embeddings_documentos.shape}")

    def recuperar_documentos(self, pregunta: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Calcula la similitud coseno entre la pregunta y la colección documental,
        devolviendo los K fragmentos más relevantes.
        """
        if self.embeddings_documentos is None or len(self.documentos) == 0:
            raise ValueError("No hay documentos indexados. Ejecuta indexar_documentos() primero.")

        embedding_pregunta = self.modelo_embeddings.encode(
            pregunta,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        puntuaciones = cosine_similarity(
            [embedding_pregunta],
            self.embeddings_documentos
        )[0]

        indices = np.argsort(puntuaciones)[::-1][:k]

        resultados = []
        for indice in indices:
            resultados.append({
                "documento": self.documentos[indice],
                "puntuacion": float(puntuaciones[indice]),
                "articulo": self.documentos[indice].get("articulo", f"Doc {indice}"),
                "seccion": self.documentos[indice].get("seccion", "General")
            })

        return resultados

    def construir_contexto(self, resultados: List[Dict[str, Any]], max_chars_por_doc: int = 450) -> str:
        """
        Ensambla los fragmentos recuperados en un contexto estructurado con citas de fuentes,
        acotando la longitud de cada fragmento para respetar la ventana de contexto del LLM.
        """
        partes = []
        for i, res in enumerate(resultados, 1):
            doc = res["documento"]
            encabezado = f"[Fuente {i}: {doc.get('seccion', '')} - {doc.get('articulo', '')}]"
            texto_doc = doc["texto"]
            if len(texto_doc) > max_chars_por_doc:
                texto_doc = texto_doc[:max_chars_por_doc] + "..."
            partes.append(f"{encabezado}\n{texto_doc}")

        return "\n\n".join(partes)

    def generar_respuesta(self, pregunta: str, contexto: str) -> str:
        """
        Envía la pregunta y el contexto estructurado al LLM con un prompt acotado.
        """
        prompt = f"""Responde a la pregunta utilizando únicamente la información del contexto.

Si el contexto no contiene la respuesta, indica exactamente:
"No encuentro esa información en los documentos disponibles."

No inventes datos ni añadas información externa.

CONTEXTO:
{contexto}

PREGUNTA:
{pregunta}

RESPUESTA:"""

        salida = self.generador(
            prompt,
            max_new_tokens=120,
            do_sample=False
        )

        return salida[0]["generated_text"].strip()

    def consultar(
        self,
        pregunta: str,
        k: int = 3,
        aplicar_umbral: bool = True
    ) -> Dict[str, Any]:
        """
        Flujo RAG completo:
        1. Recuperar K documentos más relevantes.
        2. Validar umbral de similitud (guardrail contra preguntas fuera de dominio).
        3. Construir contexto enriquecido.
        4. Generar respuesta con el modelo.
        5. Retornar paquete con respuesta, citas y metadatos.
        """
        resultados = self.recuperar_documentos(pregunta, k=k)
        max_puntuacion = resultados[0]["puntuacion"] if resultados else 0.0

        # Guardrail: Si ni el documento más afín supera el umbral mínimo,
        # evitamos alucinaciones del LLM respondiendo con fallback controlado.
        if aplicar_umbral and max_puntuacion < self.umbral_similitud:
            return {
                "pregunta": pregunta,
                "respuesta": "No encuentro esa información en los documentos disponibles.",
                "fuentes": resultados,
                "contexto": "",
                "max_puntuacion": max_puntuacion,
                "filtrado_por_umbral": True
            }

        contexto = self.construir_contexto(resultados)
        respuesta = self.generar_respuesta(pregunta, contexto)

        return {
            "pregunta": pregunta,
            "respuesta": respuesta,
            "fuentes": resultados,
            "contexto": contexto,
            "max_puntuacion": max_puntuacion,
            "filtrado_por_umbral": False
        }
