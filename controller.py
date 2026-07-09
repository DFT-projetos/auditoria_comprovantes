import subprocess
import argparse
import logging
import math
import sys
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def orquestrar_trabalhadores_por_data(data_inicio_str: str, data_fim_str: str, num_workers: int) -> None:
    dt_inicio: datetime = datetime.strptime(data_inicio_str, "%Y-%m-%d")
    dt_fim: datetime = datetime.strptime(data_fim_str, "%Y-%m-%d")
    
    if dt_fim < dt_inicio:
        logging.error("A data final não pode ser anterior à data inicial.")
        return

    dias_totais: int = (dt_fim - dt_inicio).days + 1
    dias_por_worker: int = math.ceil(dias_totais / num_workers)
    
    logging.info(f"Orquestrando {dias_totais} dias de histórico distribuídos entre {num_workers} workers.")
    logging.info(f"Cada worker será responsável por processar até {dias_por_worker} dia(s).")

    for i in range(num_workers):
        inicio_worker: datetime = dt_inicio + timedelta(days=(i * dias_por_worker))
        
        if inicio_worker > dt_fim:
            break
            
        fim_worker: datetime = min(inicio_worker + timedelta(days=dias_por_worker - 1), dt_fim)
        
        str_inicio: str = inicio_worker.strftime("%Y-%m-%d")
        str_fim: str = fim_worker.strftime("%Y-%m-%d")
        
        logging.info(f"Disparando Worker {i+1}: de {str_inicio} a {str_fim}")

        # O uso de sys.executable obriga o Windows a abrir a nova janela utilizando
        # exatamente o mesmo interpretador (e bibliotecas) que está rodando este arquivo.
        comando: str = f'start "Worker {i+1}" "{sys.executable}" worker.py {str_inicio} {str_fim}'
        subprocess.run(comando, shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orquestrador dinâmico de workers baseado em períodos.")
    parser.add_argument("--inicio", type=str, required=True, help="Data inicial (YYYY-MM-DD)")
    parser.add_argument("--fim", type=str, required=True, help="Data final (YYYY-MM-DD)")
    parser.add_argument("--workers", type=int, default=4, help="Número de instâncias paralelas")
    
    args = parser.parse_args()
    
    orquestrar_trabalhadores_por_data(args.inicio, args.fim, args.workers)