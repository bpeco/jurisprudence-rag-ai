# tools/bq_metadata.py
from google.adk.tools.tool_context import ToolContext
from google.cloud import bigquery
import os

bq = bigquery.Client()

def bq_metadata(
    bucket_pdf_link: str,
    tool_context: ToolContext,
) -> dict:
    # 1) Extrae solo el nombre del archivo (sin path ni esquema)
    print(f"[DEBUG bq_metadata] bucket_pdf_link raw: '{bucket_pdf_link}'")
    pdf_name = os.path.basename(bucket_pdf_link)
    print(f"[DEBUG bq_metadata] pdf_name raw: '{pdf_name}'")

    QUERY = """
    SELECT tribunal, expediente_n, caratula, fecha_sentencia, sala
    FROM `fallos-argentina-rag.fallos_2024_metadata.metadata_2024`
    WHERE Bucket_PDF_link = @link
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("link", "STRING", pdf_name)
        ]
    )
    job = bq.query(QUERY, job_config=job_config)
    rows = list(job)
    print(f"[DEBUG bq_metadata] row raw: '{rows}'")
    if not rows:
        return {
            "status": "no_data",
            "bucket_pdf_link": pdf_name,
            "message": f"No se encontraron metadatos para '{pdf_name}'"
        }
    row = rows[0]
    return {
        "status": "success",
        "bucket_pdf_link": pdf_name,
        "tribunal": row["tribunal"],
        "expediente_n": row["expediente_n"],
        "caratula": row["caratula"],
        "fecha_sentencia": str(row["fecha_sentencia"]),
        "sala": row["sala"],
    }
