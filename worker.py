import os
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, List

import requests
import pymysql
from dotenv import load_dotenv

from main import AnalisadorImagem, GerenciadorBancoLocal

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Worker %(process)d] %(message)s",
    handlers=[logging.StreamHandler()]
)

class TrabalhadorHistoricoData:
    def __init__(self, data_inicio: str, data_fim: str) -> None:
        self.data_inicio: str = data_inicio
        self.data_fim: str = data_fim
        self.db_local = GerenciadorBancoLocal() 
        self.limiar_foco: float = float(os.getenv("LIMIAR_FOCO_MINIMO", "100.0"))
        self.limite_lote: int = 500

    def _conectar_mysql(self) -> pymysql.connections.Connection:
        return pymysql.connect(
            host=os.getenv("DB_HOST"), port=int(os.getenv("DB_PORT", "3306")),
            user=os.getenv("DB_USER"), password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"), cursorclass=pymysql.cursors.DictCursor
        )

    def executar(self) -> None:
        logging.info(f"Iniciando processamento histórico: Período {self.data_inicio} até {self.data_fim}")
        
        ultimo_id_processado: int = 0 
        
        # Mapa de conversão para Integer (Normalização)
        mapa_status: Dict[str, int] = {
            "LEGIVEL": 1,
            "ILEGIVEL": 2,
            "ERRO_DOWNLOAD": 3,
            "ERRO": 4
        }

        while True:
            try:
                remote_conn = self._conectar_mysql()
            except Exception as e:
                logging.error(f"Erro no MySQL remoto: {e}")
                return

            query: str = """
                SELECT
                    a.id_anexo, a.id_minuta, a.imagem AS url_imagem, u.logim AS operador,
                    u.nome_completo AS nome_operador, a.`data` AS data_comprovante, a.hora AS hora_comprovante,
                    uni.sigla AS `base`, f.razao AS `cliente`
                FROM anexos a
                LEFT JOIN usuarios u ON a.operador = u.id_usuario
                LEFT JOIN minuta m ON a.id_minuta = m.id_minuta
                LEFT JOIN unidades uni ON m.unidade = uni.id_unidade
                LEFT JOIN fornecedores f ON m.id_cliente = f.id_local
                WHERE a.tipo = 1 
                  AND a.`data` >= %s AND a.`data` <= %s
                  AND a.id_anexo > %s 
                  AND a.imagem NOT LIKE '%%.pdf'
                ORDER BY a.id_anexo ASC LIMIT %s
            """

            try:
                with remote_conn.cursor() as cursor:
                    cursor.execute(query, (self.data_inicio, self.data_fim, ultimo_id_processado, self.limite_lote))
                    registros: List[Dict[str, Any]] = cursor.fetchall()

                if not registros:
                    logging.info("Nenhum registro pendente para este período. Finalizando Worker.")
                    break

                logging.info(f"Processando bloco de {len(registros)} registros (A partir do ID {registros[0]['id_anexo']})...")

                for reg in registros:
                    id_anexo: int = int(reg['id_anexo'])
                    url: str = reg['url_imagem']
                    status_texto: str = "ERRO_DOWNLOAD"
                    foco_medido: float = 0.0

                    if url:
                        try:
                            resp = requests.get(url, timeout=10)
                            if resp.status_code == 200:
                                analise = AnalisadorImagem.analisar(resp.content, self.limiar_foco)
                                status_texto = analise["status"]
                                foco_medido = analise["foco"]
                        except requests.RequestException:
                            pass

                    id_status: int = mapa_status.get(status_texto, 4)

                    self.db_local.salvar_resultado({
                        "id_anexo": id_anexo, "id_minuta": reg['id_minuta'],
                        "operador": reg['operador'], "nome_operador": reg['nome_operador'],
                        "base": reg['base'], "cliente": reg['cliente'],
                        "url_imagem": url, "id_status_legibilidade": id_status,
                        "foco": foco_medido, "data_comprovante": str(reg['data_comprovante']),
                        "hora_comprovante": str(reg['hora_comprovante']),
                        "processado_em": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    ultimo_id_processado = id_anexo

            finally:
                remote_conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processador de histórico de comprovantes por Data.")
    parser.add_argument("data_inicio", type=str, help="Data Inicial (Formato YYYY-MM-DD)")
    parser.add_argument("data_fim", type=str, help="Data Final (Formato YYYY-MM-DD)")
    
    args = parser.parse_args()
    trabalhador = TrabalhadorHistoricoData(args.data_inicio, args.data_fim)
    trabalhador.executar()
