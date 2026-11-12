"""
Aplicación Principal / CLI para el Asistente Documental de Metro Bilbao.
Permite ejecutar consultas interactivas o correr la batería de pruebas y evaluación.
"""

import os
import sys

# Detección y auto-activación automática de .venv si se ejecuta con el python del sistema
try:
    import torch
except ImportError:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    venv_py = os.path.join(base_dir, ".venv", "bin", "python")
    if os.path.exists(venv_py) and sys.executable != venv_py:
        os.execv(venv_py, [venv_py] + sys.argv)
    else:
        print("[ERROR] No se encontró 'torch'. Asegúrate de activar el entorno virtual:")
        print("        source .venv/bin/activate")
        sys.exit(1)

import argparse

# Asegurar importación de src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.document_loader import cargar_documentos_por_articulos
from src.rag_pipeline import MetroBilbaoRAG
from src.evaluation import CONJUNTO_EVALUACION, evaluar_asistente


def imprimir_encabezado():
    print("=" * 75)
    print("ASISTENTE DOCUMENTAL INTELIGENTE - METRO BILBAO (RAG)")
    print("Sistema RAG basado en las Condiciones Generales de Contratación")
    print("Motor de Embeddings: BGE (BAAI/bge-m3)")
    print("=" * 75)


def modo_interactivo(rag):
    print("\n[INFO] Modo Interactivo activado. Escribe tus preguntas sobre la normativa de Metro Bilbao.")
    print("       (Escribe 'salir', 'exit' o 'q' para finalizar)\n")

    while True:
        try:
            pregunta = input("\nTu pregunta: ").strip()
            if not pregunta:
                continue
            if pregunta.lower() in ["salir", "exit", "q"]:
                print("Hasta luego. Buen viaje en Metro Bilbao.")
                break

            print("\nBuscando normativa aplicable y generando respuesta...")
            resultado = rag.consultar(pregunta, k=3)

            print("\n" + "-" * 60)
            print("RESPUESTA DEL ASISTENTE:")
            print(resultado["respuesta"])
            print("-" * 60)

            print("FUENTES Y ARTICULOS CONSULTADOS:")
            for i, f in enumerate(resultado["fuentes"], 1):
                puntos = f["puntuacion"]
                art = f["articulo"]
                sec = f["seccion"]
                print(f"  {i}. [{puntos:.4f}] {sec} -> {art}")

            if resultado.get("filtrado_por_umbral"):
                print("  [AVISO] Nota: Pregunta filtrada por bajo umbral de similitud (fuera de dominio).")

        except KeyboardInterrupt:
            print("\nInterrupción detectada. Cerrando asistente.")
            break


def ejecutar_pruebas(rag):
    print("\n[INFO] Ejecutando batería de pruebas oficiales (7 preguntas)...")
    df = evaluar_asistente(rag, k=3)

    print("\n" + "=" * 90)
    print("RESULTADOS DE LA EVALUACIÓN:")
    print("=" * 90)
    cols = ["id", "pregunta", "tipo", "top1_recuperado", "score_similitud", "recuperacion (1-5)"]
    print(df[cols].to_string(index=False))

    print("\n" + "=" * 90)
    print("DETALLE DE RESPUESTAS GENERADAS:")
    print("=" * 90)
    for _, fila in df.iterrows():
        print(f"\n[ID {fila['id']}] ({fila['tipo']}) {fila['pregunta']}")
        print(f"-> Top 1: {fila['top1_recuperado']} (Score: {fila['score_similitud']})")
        print(f"-> Respuesta: {fila['respuesta_generada']}")
        print(f"-> Observaciones: {fila['observaciones']}")


def main():
    parser = argparse.ArgumentParser(description="Asistente RAG de Metro Bilbao (Exclusivo con modelo BGE)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Iniciar en modo chat interactivo")
    parser.add_argument("--eval", "-e", action="store_true", help="Ejecutar evaluación automática de pruebas")
    parser.add_argument(
        "--embedding",
        default="BAAI/bge-m3",
        help="Modelo de embeddings BGE (por defecto: BAAI/bge-m3. Solo se permiten modelos de la familia BGE)"
    )
    parser.add_argument(
        "--generador",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Modelo generativo LLM (por defecto: Qwen/Qwen2.5-3B-Instruct. Compatible con Qwen2.5, Flan-T5, Llama, etc.)"
    )
    args = parser.parse_args()

    # Validación estricta: el proyecto solo funciona con el modelo BGE
    if "bge" not in args.embedding.lower():
        parser.error(
            f"El modelo de embeddings '{args.embedding}' no está permitido.\n"
            f"Este proyecto está configurado para operar exclusivamente con la familia de modelos BGE (ej: 'BAAI/bge-m3')."
        )

    imprimir_encabezado()

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "condiciones_generales_metro_bilbao.md")
    print(f"\n[1/3] Cargando base documental desde: {data_path}")
    docs = cargar_documentos_por_articulos(data_path)
    print(f"      Total de artículos cargados: {len(docs)}")

    print(f"\n[2/3] Inicializando modelos (Embedding: {args.embedding} | LLM: {args.generador})...")
    rag = MetroBilbaoRAG(
        embedding_model_name=args.embedding,
        generation_model_name=args.generador
    )

    print("\n[3/3] Indexando base documental...")
    rag.indexar_documentos(docs)

    if args.interactive:
        modo_interactivo(rag)
    elif args.eval:
        ejecutar_pruebas(rag)
    else:
        # Por defecto, ejecuta las pruebas y luego ofrece el modo interactivo
        ejecutar_pruebas(rag)
        modo_interactivo(rag)


if __name__ == "__main__":
    main()
