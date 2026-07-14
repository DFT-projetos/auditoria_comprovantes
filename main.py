import io
import os
import time
import sqlite3
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import requests
import numpy as np
import cv2
import pymysql
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)


class GerenciadorBancoLocal:
    """Encapsula todas as operações do SQLite local (Tipagem Estrita)."""

    def __init__(self, db_name: str = "auditoria_comprovantes.db") -> None:
        self.db_name: str = db_name
        self._inicializar_banco()

    def _inicializar_banco(self) -> None:
        with sqlite3.connect(self.db_name, timeout=15.0) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ponteiro_controle (
                    chave TEXT PRIMARY KEY,
                    ultimo_id INTEGER NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS resultados_analise (
                    id_anexo INTEGER PRIMARY KEY,
                    id_minuta INTEGER,
                    operador TEXT,
                    nome_operador TEXT,
                    base TEXT,
                    cliente TEXT,
                    url_imagem TEXT,
                    id_status_legibilidade INTEGER,
                    foco REAL,
                    data_comprovante TEXT,
                    hora_comprovante TEXT,
                    processado_em TEXT,
                    enviado INTEGER DEFAULT 0
                )
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_nao_enviados ON resultados_analise(enviado);")
            cursor.execute(
                "INSERT OR IGNORE INTO ponteiro_controle (chave, ultimo_id) VALUES ('ultimo_id_anexo', 0)")
            conn.commit()

    def obter_ultimo_id(self) -> int:
        with sqlite3.connect(self.db_name, timeout=15.0) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT ultimo_id FROM ponteiro_controle WHERE chave = 'ultimo_id_anexo'")
            resultado: Optional[Tuple[int]] = cursor.fetchone()
            return int(resultado[0]) if resultado else 0

    def atualizar_ultimo_id(self, novo_id: int) -> None:
        with sqlite3.connect(self.db_name, timeout=15.0) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE ponteiro_controle SET ultimo_id = ? WHERE chave = 'ultimo_id_anexo'", (novo_id,))
            conn.commit()

    def salvar_resultado(self, dados: Dict[str, Any]) -> None:
        with sqlite3.connect(self.db_name, timeout=15.0) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO resultados_analise 
                (id_anexo, id_minuta, operador, nome_operador, base, cliente, url_imagem, id_status_legibilidade, foco, data_comprovante, hora_comprovante, processado_em)
                VALUES (:id_anexo, :id_minuta, :operador, :nome_operador, :base, :cliente, :url_imagem, :id_status_legibilidade, :foco, :data_comprovante, :hora_comprovante, :processado_em)
            """, dados)
            conn.commit()

    def obter_nao_enviados(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_name, timeout=15.0) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM resultados_analise WHERE enviado = 0")
            return [dict(row) for row in cursor.fetchall()]

    def marcar_enviado(self, id_anexo: int) -> None:
        with sqlite3.connect(self.db_name, timeout=15.0) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE resultados_analise SET enviado = 1 WHERE id_anexo = ?", (id_anexo,))
            conn.commit()


class AnalisadorImagem:
    """Classe utilitária para o processamento de visão computacional."""

    @staticmethod
    def _calcular_caracteristicas(img: np.ndarray) -> Dict[str, float]:
        cinza: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        foco_medido: float = float(cv2.Laplacian(cinza, cv2.CV_64F).var())

        suave: np.ndarray = cv2.GaussianBlur(cinza, (5, 5), 0)
        bordas: np.ndarray = cv2.Canny(suave, 75, 200)
        densidade_bordas: float = float(np.count_nonzero(bordas)) / float(bordas.size)
        contraste: float = float(cinza.std())

        hist = cv2.calcHist([cinza], [0], None, [256], [0, 256])
        hist = hist / max(hist.sum(), 1.0)
        entropia: float = float(-(hist[hist > 0] * np.log2(hist[hist > 0])).sum()) if np.any(hist > 0) else 0.0

        return {
            "foco": foco_medido,
            "densidade_bordas": densidade_bordas,
            "contraste": contraste,
            "entropia": entropia,
        }

    @staticmethod
    def _configurar_ocr_local() -> Optional[Any]:
        try:
            import pytesseract
        except Exception:
            return None

        caminho_tesseract = os.getenv("TESSERACT_CMD") or os.getenv("TESSERACT_PATH")
        if caminho_tesseract:
            pytesseract.pytesseract.tesseract_cmd = caminho_tesseract

        try:
            pytesseract.pytesseract.tesseract_version()
            return pytesseract
        except Exception:
            return None

    @staticmethod
    def _executar_ocr_local(conteudo_bytes: bytes, img: np.ndarray) -> Dict[str, Any]:
        pytesseract = AnalisadorImagem._configurar_ocr_local()
        if pytesseract is None:
            return {"status": "indisponivel", "texto": "", "palavras": 0}

        try:
            from PIL import Image
        except Exception:
            return {"status": "indisponivel", "texto": "", "palavras": 0}

        imagem_pil = Image.open(io.BytesIO(conteudo_bytes)).convert("L")
        largura, altura = imagem_pil.size
        if largura < 300 or altura < 300:
            imagem_pil = imagem_pil.resize((max(largura * 2, 600), max(altura * 2, 600)))

        imagem_preprocessada = imagem_pil.point(lambda p: 255 if p > 180 else 0, "1")
        dados = pytesseract.image_to_data(
            imagem_preprocessada,
            output_type=pytesseract.Output.DICT,
            config="--psm 6 --oem 3"
        )
        textos = [texto for texto in dados.get("text", []) if texto and texto.strip()]
        return {"status": "ok", "texto": " ".join(textos), "palavras": len(textos)}

    @staticmethod
    def _verificar_secundariamente(conteudo_bytes: bytes, img: np.ndarray, limiar_foco: float) -> Dict[str, Any]:
        if img is None:
            return {"aplicado": False, "status": "ERRO", "motivo": "Imagem indisponível para verificação secundária.", "metodo": "nenhum"}

        if os.getenv("ENABLE_SEGUNDA_VERIFICACAO", "1").lower() not in {"1", "true", "yes", "on"}:
            return {
                "aplicado": False,
                "status": "PULADO",
                "motivo": "Segunda verificação desativada por configuração.",
                "metodo": "desativado",
            }

        caracteristicas = AnalisadorImagem._calcular_caracteristicas(img)

        heuristica_legivel = (
            caracteristicas["foco"] >= max(limiar_foco * 0.65, 30.0)
            and caracteristicas["densidade_bordas"] >= 0.01
            and caracteristicas["contraste"] >= 20.0
        )

        if heuristica_legivel:
            return {
                "aplicado": True,
                "status": "LEGIVEL",
                "motivo": "Verificação secundária heurística aprovou a imagem como potencialmente legível.",
                "metodo": "heuristica",
                "foco": round(caracteristicas["foco"], 2),
            }

        ocr = AnalisadorImagem._executar_ocr_local(conteudo_bytes, img)
        if ocr.get("status") == "ok" and ocr.get("palavras", 0) >= 3:
            return {
                "aplicado": True,
                "status": "LEGIVEL",
                "motivo": "OCR local encontrou texto suficiente para indicar legibilidade.",
                "metodo": "ocr_local",
                "foco": round(caracteristicas["foco"], 2),
                "texto_ocr": ocr.get("texto", ""),
            }

        return {
            "aplicado": True,
            "status": "ILEGIVEL",
            "motivo": "Verificação secundária local não encontrou sinais de legibilidade suficientes.",
            "metodo": "ocr_local" if ocr.get("status") == "ok" else "heuristica",
            "foco": round(caracteristicas["foco"], 2),
            "texto_ocr": ocr.get("texto", ""),
        }

    @staticmethod
    def analisar(conteudo_bytes: bytes, limiar_foco: float = 100.0) -> Dict[str, Any]:
        try:
            image_array: np.ndarray = np.asarray(
                bytearray(conteudo_bytes), dtype=np.uint8)
            img: np.ndarray = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if img is None:
                return {"status": "ERRO", "motivo": "Falha ao decodificar imagem", "foco": 0.0}

            cinza: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            foco_medido: float = float(cv2.Laplacian(cinza, cv2.CV_64F).var())
            status_primario: str = "LEGIVEL"
            motivo_primario: str = f"Documento nítido (Foco: {foco_medido:.1f})"

            if foco_medido < limiar_foco:
                status_primario = "ILEGIVEL"
                motivo_primario = f"Imagem borrada (Foco: {foco_medido:.1f})"
            else:
                suave: np.ndarray = cv2.GaussianBlur(cinza, (5, 5), 0)
                bordas: np.ndarray = cv2.Canny(suave, 75, 200)
                kernel: np.ndarray = cv2.getStructuringElement(
                    cv2.MORPH_RECT, (30, 30))
                bordas_dilatadas: np.ndarray = cv2.dilate(
                    bordas, kernel, iterations=2)

                contornos, _ = cv2.findContours(
                    bordas_dilatadas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if not contornos:
                    status_primario = "ILEGIVEL"
                    motivo_primario = "Nenhum documento detectado."
                else:
                    maior_contorno: np.ndarray = max(contornos, key=cv2.contourArea)
                    _, _, w, h = cv2.boundingRect(maior_contorno)
                    proporcao_area: float = float(
                        w * h) / float(img.shape[0] * img.shape[1])

                    if proporcao_area < 0.15:
                        status_primario = "ILEGIVEL"
                        motivo_primario = f"Área insuficiente ({proporcao_area:.1%})"

            if status_primario == "LEGIVEL":
                return {"status": "LEGIVEL", "motivo": motivo_primario, "foco": round(foco_medido, 2)}

            verificacao_secundaria = AnalisadorImagem._verificar_secundariamente(conteudo_bytes, img, limiar_foco)
            if verificacao_secundaria.get("status") == "LEGIVEL":
                return {
                    "status": "LEGIVEL",
                    "motivo": verificacao_secundaria["motivo"],
                    "foco": round(foco_medido, 2),
                    "segunda_verificacao": verificacao_secundaria,
                }

            return {
                "status": "ILEGIVEL",
                "motivo": f"{motivo_primario} | {verificacao_secundaria['motivo']}",
                "foco": round(foco_medido, 2),
                "segunda_verificacao": verificacao_secundaria,
            }

        except Exception as e:
            return {"status": "ERRO", "motivo": str(e), "foco": 0.0}


class PipelineAuditoria:
    """Orquestra a extração, análise e sincronização dos dados."""

    def __init__(self) -> None:
        self.db_local = GerenciadorBancoLocal()
        self.url_api: str = os.getenv(
            "URL_API_SAIDA", "http://localhost:3001/api/comprovantes")
        self.limiar_foco: float = float(
            os.getenv("LIMIAR_FOCO_MINIMO", "100.0"))
        self.limite_lote: int = int(os.getenv("LIMITE_LOTE", "20"))
        self.data_corte: str = os.getenv("DATA_INICIAL_CORTE", "2026-01-01")

    def _conectar_mysql(self) -> pymysql.connections.Connection:
        return pymysql.connect(
            host=os.getenv("DB_HOST"), port=int(os.getenv("DB_PORT", "3306")),
            user=os.getenv("DB_USER"), password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"), cursorclass=pymysql.cursors.DictCursor
        )

    def extrair_e_processar(self) -> None:
        ultimo_id: int = self.db_local.obter_ultimo_id()

        mapa_status: Dict[str, int] = {
            "LEGIVEL": 1,
            "ILEGIVEL": 2,
            "ERRO_DOWNLOAD": 3,
            "ERRO": 4
        }

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
            WHERE a.tipo = 1 AND a.`data` >= %s AND a.id_anexo > %s AND a.imagem NOT LIKE '%%.pdf'
            ORDER BY a.id_anexo ASC LIMIT %s
        """

        try:
            with remote_conn.cursor() as cursor:
                cursor.execute(
                    query, (self.data_corte, ultimo_id, self.limite_lote))
                registros: List[Dict[str, Any]] = cursor.fetchall()

            if not registros:
                return

            for reg in registros:
                id_anexo: int = int(reg['id_anexo'])
                url: str = reg['url_imagem']
                status_texto: str = "ERRO_DOWNLOAD"
                foco_medido: float = 0.0

                if url:
                    try:
                        resp = requests.get(url, timeout=15)
                        if resp.status_code == 200:
                            analise = AnalisadorImagem.analisar(
                                resp.content, self.limiar_foco)
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
                self.db_local.atualizar_ultimo_id(id_anexo)
        finally:
            remote_conn.close()

    def sincronizar_api(self) -> None:
        pendentes: List[Dict[str, Any]] = self.db_local.obter_nao_enviados()
        if not pendentes:
            return

        logging.info(f"Sincronização: {len(pendentes)} registros pendentes.")
        for reg in pendentes:
            try:
                resp = requests.post(self.url_api, json=reg, timeout=10)
                if resp.status_code in (200, 201):
                    self.db_local.marcar_enviado(int(reg["id_anexo"]))
                else:
                    logging.warning(
                        f"API retornou {resp.status_code} para ID {reg['id_anexo']}")
            except requests.RequestException as e:
                logging.error(
                    f"Erro de rede ao sincronizar ID {reg['id_anexo']}: {e}")
                break

    def loop_principal(self) -> None:
        intervalo: int = int(os.getenv("INTERVALO_SEGUNDOS", "30"))
        logging.info("=== Serviço Iniciado ===")
        while True:
            try:
                self.extrair_e_processar()
                self.sincronizar_api()
            except Exception as e:
                logging.error(f"Falha crítica no loop: {e}")
            time.sleep(intervalo)


if __name__ == "__main__":
    app = PipelineAuditoria()
    app.loop_principal()
