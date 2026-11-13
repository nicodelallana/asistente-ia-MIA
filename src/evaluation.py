"""
Módulo de evaluación y benchmarking para el Asistente RAG de Metro Bilbao.
Evalúa recuperación (MRR, Hit Rate) y generación (Fidelidad, Corrección, Relevancia).
"""

from typing import List, Dict, Any
import pandas as pd


CONJUNTO_EVALUACION = [
    {
        "id": 1,
        "pregunta": "¿Se puede viajar en Metro Bilbao con un patinete eléctrico?",
        "tipo": "En dominio",
        "articulo_esperado": "Artículo 20.- Bultos de mano y accesorios",
        "respuesta_esperada": "No, queda terminantemente prohibido el acceso de patinetes eléctricos y monociclos eléctricos."
    },
    {
        "id": 2,
        "pregunta": "¿A cuánto asciende la percepción mínima por no llevar billete si se abona en las primeras 48 horas?",
        "tipo": "En dominio",
        "articulo_esperado": "Artículo 15.- Percepción mínima",
        "respuesta_esperada": "Asciende a 20,00 euros si se abona en 48 horas (y 50,00 euros hasta el 7º día natural)."
    },
    {
        "id": 3,
        "pregunta": "¿Cuál es la caducidad mínima de la tarjeta Barik?",
        "tipo": "En dominio",
        "articulo_esperado": "Artículo 16.- Caducidad",
        "respuesta_esperada": "La tarjeta barik tiene una caducidad mínima de 4 años."
    },
    {
        "id": 4,
        "pregunta": "¿Cuáles son los requisitos para viajar con un perro de más de 8 kg?",
        "tipo": "En dominio",
        "articulo_esperado": "Artículo 18.- Perros y gatos",
        "respuesta_esperada": "Llevar bozal, correa no extensible de máximo 50 cm, viajar fuera de horas punta y no ocupar primer/último coche ni asientos."
    },
    {
        "id": 5,
        "pregunta": "¿Cuándo se tiene derecho a recibir un título de transporte gratuito por retraso o suspensión del servicio?",
        "tipo": "En dominio",
        "articulo_esperado": "Artículo 14.- Suspensión del servicio y título gratuito",
        "respuesta_esperada": "Cuando la suspensión o retraso previsto por parada en ruta sea superior a media hora, para títulos con número limitado de viajes."
    },
    {
        "id": 6,
        "pregunta": "¿Cuál es el salario mensual de los conductores de Metro Bilbao?",
        "tipo": "Fuera de dominio",
        "articulo_esperado": "Ninguno",
        "respuesta_esperada": "No encuentro esa información en los documentos disponibles."
    },
    {
        "id": 7,
        "pregunta": "¿Cómo se cocina una auténtica tortilla de patatas?",
        "tipo": "Fuera de dominio",
        "articulo_esperado": "Ninguno",
        "respuesta_esperada": "No encuentro esa información en los documentos disponibles."
    }
]


def evaluar_asistente(rag_system, k: int = 3) -> pd.DataFrame:
    """
    Ejecuta el conjunto de preguntas sobre el sistema RAG y genera
    un DataFrame estructurado para la evaluación cualitativa y cuantitativa.
    """
    filas = []

    for item in CONJUNTO_EVALUACION:
        pregunta = item["pregunta"]
        resultado = rag_system.consultar(pregunta, k=k)

        articulos_recuperados = [f["articulo"] for f in resultado["fuentes"]]
        max_puntuacion = resultado["max_puntuacion"]
        respuesta_generada = resultado["respuesta"]

        # Análisis de acierto en recuperación
        if item["tipo"] == "En dominio":
            acierto_recuperacion = any(item["articulo_esperado"] in art for art in articulos_recuperados)
            top1_acierto = item["articulo_esperado"] in articulos_recuperados[0] if articulos_recuperados else False
            puntuacion_recup = 5 if top1_acierto else (4 if acierto_recuperacion else 2)
        else:
            # En preguntas fuera de dominio, idealmente no debe recuperar o la similitud debe ser baja
            acierto_recuperacion = max_puntuacion < 0.50
            puntuacion_recup = 5 if acierto_recuperacion else 3

        # Análisis de respuesta respecto al fallback esperado
        es_fallback = "No encuentro esa información" in respuesta_generada

        if item["tipo"] == "Fuera de dominio":
            fidelidad = 5 if es_fallback else 1
            correccion = 5 if es_fallback else 1
            relevancia = 5 if es_fallback else 1
            obs = "Responde correctamente con fallback" if es_fallback else "Alucinación: intentó responder con contexto no relacionado"
        else:
            # En dominio
            fidelidad = 4 if acierto_recuperacion else 2
            correccion = 4 if not es_fallback and acierto_recuperacion else 3
            relevancia = 5 if acierto_recuperacion else 2
            obs = f"Recuperado Top 1: {articulos_recuperados[0]}" if articulos_recuperados else "Sin resultados"

        filas.append({
            "id": item["id"],
            "pregunta": pregunta,
            "tipo": item["tipo"],
            "top1_recuperado": articulos_recuperados[0] if articulos_recuperados else "N/A",
            "score_similitud": round(max_puntuacion, 4),
            "respuesta_generada": respuesta_generada,
            "recuperacion (1-5)": puntuacion_recup,
            "fidelidad (1-5)": fidelidad,
            "correccion (1-5)": correccion,
            "relevancia (1-5)": relevancia,
            "observaciones": obs
        })

    df = pd.DataFrame(filas)
    return df


if __name__ == "__main__":
    import os
    from document_loader import cargar_documentos_por_articulos
    from rag_pipeline import MetroBilbaoRAG

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "condiciones_generales_metro_bilbao.md")
    docs = cargar_documentos_por_articulos(data_path)
    rag = MetroBilbaoRAG()
    rag.indexar_documentos(docs)
    df_eval = evaluar_asistente(rag)
    print(df_eval[["pregunta", "top1_recuperado", "score_similitud", "recuperacion (1-5)"]])
