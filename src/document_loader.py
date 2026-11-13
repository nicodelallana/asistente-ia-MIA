"""
Módulo para la carga y fragmentación de la base documental de Metro Bilbao.
"""

import os
import re
from typing import List, Dict, Any


def cargar_documentos_por_articulos(filepath: str) -> List[Dict[str, Any]]:
    """
    Carga el archivo Markdown de las Condiciones Generales de Metro Bilbao
    y lo divide semánticamente a nivel de Artículo (3 al 30), conservando
    la sección padre como metadato.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró el archivo documental en: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    documentos = []
    seccion_actual = "Condiciones Generales"
    doc_id = 0
    articulo_actual = None
    texto_actual = []

    def flush():
        nonlocal doc_id, articulo_actual, texto_actual
        if articulo_actual is not None and texto_actual:
            cuerpo = "".join(texto_actual).strip()
            # Formato enriquecido para embedding que incluye título y contexto
            texto_enriquecido = f"[{seccion_actual} | {articulo_actual}]\n{cuerpo}"
            documentos.append({
                "id": doc_id,
                "seccion": seccion_actual,
                "articulo": articulo_actual,
                "texto": cuerpo,
                "texto_completo": texto_enriquecido,
                "fuente": os.path.basename(filepath)
            })
            doc_id += 1
            articulo_actual = None
            texto_actual = []

    for line in lines:
        if line.startswith("## "):
            flush()
            seccion_actual = line.strip().replace("## ", "").strip()
        elif line.startswith("### Artículo "):
            flush()
            articulo_actual = line.strip().replace("### ", "").strip()
        elif articulo_actual is not None:
            texto_actual.append(line)

    flush()
    return documentos


def cargar_documentos_fijos(filepath: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict[str, Any]]:
    """
    Estrategia de fragmentación alternativa (chunks fijos por caracteres con solapamiento),
    útil para comparar contra la segmentación semántica por artículos.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        texto = f.read()

    chunks = []
    inicio = 0
    doc_id = 0
    while inicio < len(texto):
        fin = inicio + chunk_size
        chunk_text = texto[inicio:fin].strip()
        if chunk_text:
            chunks.append({
                "id": doc_id,
                "seccion": "Chunk Fijo",
                "articulo": f"Segmento {doc_id + 1}",
                "texto": chunk_text,
                "texto_completo": chunk_text,
                "fuente": os.path.basename(filepath)
            })
            doc_id += 1
        inicio += (chunk_size - overlap)

    return chunks


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_path = os.path.join(base_dir, "data", "condiciones_generales_metro_bilbao.md")
    docs = cargar_documentos_por_articulos(default_path)
    print(f"Cargados {len(docs)} artículos con éxito.")
    for d in docs[:3]:
        print(f"- [{d['id']}] {d['articulo']} ({len(d['texto'])} caracteres)")
