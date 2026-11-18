import argparse
import io
import json
import os
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
from PIL import Image, ImageOps
from pydantic import BaseModel, Field, model_validator

# --- 1. Caminhos Seguros e Configuração de Ambiente ---
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
PASTA_PROCESSADAS = BASE_DIR / "inspecoes_processadas"
EXTENSOES_ACEITAS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# --- 2. Contratos de Dados (Pydantic Schemas) ---
class DimensoesEscalaRegua(BaseModel):
    comprimento_cm: Optional[float] = Field(
        None, ge=0, description="Comprimento estimado da peça em centímetros, calculado usando a régua de 30cm como base."
    )
    largura_cm: Optional[float] = Field(
        None, ge=0, description="Largura estimada da peça em centímetros, calculada usando a régua de 30cm como base."
    )
    regua_30cm_detectada: bool = Field(
        description="True se a régua padrão de 30cm foi encontrada na imagem para calibração métrica."
    )
    observacao_escala: str = Field(
        description="Breve relato sobre a leitura da régua e a proporção aplicada."
    )

class EstatisticasFendas(BaseModel):
    quantidade: int = Field(
        ge=0,
        description="Número total de fendas ou rachaduras visíveis.",
    )
    severidade_maxima: Literal[
        "nenhuma",
        "superficial_fina",
        "moderada",
        "estrutural_passante",
    ]
    localizacao: Literal[
        "nenhuma",
        "topo_extremidade",
        "centro_face",
        "bordas",
        "multiplas_regioes",
    ]
    extensao_relativa_percentual: float = Field(
        ge=0,
        le=100,
        description="Percentual do comprimento da peça afetado pela maior fenda (0 a 100).",
    )


class EstatisticasNos(BaseModel):
    nos_firmes: int = Field(
        ge=0,
        description="Nós sadios, claros e integrados à fibra da madeira.",
    )
    nos_soltos_ou_cariados: int = Field(
        ge=0,
        description="Nós escuros, soltos, rachados, podres ou com perda de material.",
    )
    quantidade_total: int = Field(
        ge=0,
        description="Soma exata de nos_firmes + nos_soltos_ou_cariados.",
    )
    area_ocupada_estimada_percentual: float = Field(
        ge=0,
        le=100,
        description="Percentual aproximado da área visível da face ocupada por nós (0 a 100).",
    )

    @model_validator(mode="after")
    def corrigir_soma_nos(self) -> "EstatisticasNos":
        self.quantidade_total = self.nos_firmes + self.nos_soltos_ou_cariados
        return self


class AnaliseEmpenamento(BaseModel):
    status: Literal[
        "nao_detectado",
        "leve",
        "moderado",
        "severo",
        "inconclusivo_angulo_foto",
    ]
    tipo_predominante: Literal[
        "nenhum",
        "encurvamento_lateral",
        "arqueamento_longitudinal",
        "encanoamento_transversal",
        "torcao",
        "indeterminado_sem_vista_topo",
    ]
    observacao_geometria: str = Field(
        description="Descrição técnica curta da deformação ou aviso sobre o ângulo da foto.",
    )


class LaudoTecnicoMadeira(BaseModel):
    qualidade_imagem_suficiente: bool = Field(
        description="True se a foto permite inspecionar a madeira; False se inadequada.",
    )
    dimensoes: DimensoesEscalaRegua
    fendas: EstatisticasFendas
    nos: EstatisticasNos
    empenamento: AnaliseEmpenamento
    parecer_tecnico_inicial: str = Field(
        description="Parecer técnico direto e objetivo em até 3 frases, sem saudações.",
    )
    conformidade: Literal[
        "CONFORME",
        "CONFORME_COM_RESSALVAS",
        "NAO_CONFORME",
        "INCONCLUSIVO_FOTO_INADEQUADA",
    ]


class RespostaBackend(BaseModel):
    sucesso: bool
    timestamp_iso: str
    arquivos_originais: list[str]
    arquivos_renomeados: list[str]
    modelo_utilizado: Optional[str] = None
    tempo_processamento_segundos: float
    laudo: Optional[LaudoTecnicoMadeira] = None
    erro: Optional[str] = None


# --- 3. Regras de Negócio e Segurança ---

SYSTEM_PROMPT = """
Você é um inspetor técnico sênior de qualidade de madeira serrada e beneficiada.
Analise as imagens fornecidas buscando não conformidades físicas (fendas, nós, empenamento) E UTILIZANDO A RÉGUA DE 30CM COMO BASE DE ESCALA.

Regras técnicas obrigatórias:
1. **Calibração por Régua Física (30cm):** Localize a régua padrão de 30 cm posicionada na imagem. Utilize as marcações dela como referência absoluta de escala para calcular com precisão métrica o comprimento e a largura reais da peça. Se a régua não for encontrada, defina `regua_30cm_detectada=false`.
2. Não confunda veios naturais escuros, variações de cerne/alburno ou marcas normais de serra com fendas.
3. Fendas têm abertura física ou sombra de profundidade longitudinal.
4. Empenamento (arqueamento, encanoamento, torção) exige visão de borda/topo ou referência de superfície plana. Se houver apenas uma foto frontal chapada sem curvatura lateral visível, defina status como 'inconclusivo_angulo_foto' e tipo como 'indeterminado_sem_vista_topo'.
5. Se a imagem estiver muito tremida, escura ou sem a régua de referência quando exigida, marque qualidade_imagem_suficiente=False e conformidade='INCONCLUSIVO_FOTO_INADEQUADA'.
"""


def obter_api_key_segura() -> str:
    """Carrega a chave exclusivamente do arquivo .env e bloqueia execução sem chave."""
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    chave = os.getenv("GEMINI_API_KEY", "").strip()

    if not chave or chave == "COLE_SUA_API_KEY_AQUI":
        if not ENV_PATH.exists():
            ENV_PATH.write_text("GEMINI_API_KEY=COLE_SUA_API_KEY_AQUI\n", encoding="utf-8")
        raise PermissionError(
            f"GEMINI_API_KEY ausente ou inválida. Configure sua chave no arquivo: {ENV_PATH}"
        )
    return chave


def localizar_imagens_para_processar(caminhos_informados: list[str]) -> list[Path]:
    """Localiza as imagens passadas por argumento ou busca automaticamente na pasta atual."""
    if caminhos_informados:
        arquivos = [Path(c).resolve() for c in caminhos_informados]
        for arq in arquivos:
            if not arq.exists():
                raise FileNotFoundError(f"Arquivo não encontrado: {arq}")
        return arquivos

    candidatos = [
        p for p in BASE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSOES_ACEITAS
    ]
    if not candidatos:
        raise FileNotFoundError(
            f"Nenhuma imagem ({', '.join(EXTENSOES_ACEITAS)}) encontrada em {BASE_DIR}"
        )

    candidatos.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [candidatos[0]]


def renomear_e_padronizar_imagem(
    caminho_origem: Path,
    indice: int,
    lote_id: str,
    max_lado: int = 1024,
) -> tuple[Path, types.Part]:
    PASTA_PROCESSADAS.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    novo_nome = f"peca_{timestamp}_{lote_id}_vista{indice}.jpg"
    destino_renomeado = PASTA_PROCESSADAS / novo_nome

    with Image.open(caminho_origem) as img:
        img_corrigida = ImageOps.exif_transpose(img).convert("RGB")
        img_corrigida.save(destino_renomeado, format="JPEG", quality=92)
        img_corrigida.thumbnail((max_lado, max_lado), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img_corrigida.save(buffer, format="JPEG", quality=85)
        bytes_api = buffer.getvalue()

    if caminho_origem.parent == BASE_DIR and caminho_origem != destino_renomeado:
        try:
            caminho_origem.unlink()
        except OSError:
            shutil.move(str(caminho_origem), str(destino_renomeado))

    part = types.Part.from_bytes(data=bytes_api, mime_type="image/jpeg")
    return destino_renomeado, part


def listar_modelos_candidatos(client: genai.Client) -> list[str]:
    modelo_fixo = os.getenv("GEMINI_MODEL", "").strip()
    if modelo_fixo:
        return [modelo_fixo]

    ignorados = (
        "tts", "image", "embedding", "veo", "imagen",
        "audio", "robotics", "thinking", "gemma", "aqa",
    )
    disponiveis = []
    try:
        for m in client.models.list():
            nome = m.name.replace("models/", "")
            if nome.startswith("gemini-") and not any(t in nome for t in ignorados):
                disponiveis.append(nome)
    except Exception:
        pass

    preferidos = [
        "gemini-3.8-flash-lite",
        "gemini-3.8-flash",
        "gemini-3.8-pro",
        "gemini-3.0-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro",
    ]
    fila = [m for m in preferidos if m in disponiveis] + [
        m for m in disponiveis if m not in preferidos
    ]
    return fila if fila else preferidos


def executar_inspecao(caminhos_entrada: list[str]) -> RespostaBackend:
    inicio = time.time()
    lote_id = uuid.uuid4().hex[:6]
    originais_str: list[str] = []
    renomeados_str: list[str] = []

    try:
        api_key = obter_api_key_segura()
        arquivos = localizar_imagens_para_processar(caminhos_entrada)
        originais_str = [arq.name for arq in arquivos]

        parts_imagens: list[types.Part] = []
        for idx, arq in enumerate(arquivos, start=1):
            caminho_novo, part = renomear_e_padronizar_imagem(arq, idx, lote_id)
            renomeados_str.append(str(caminho_novo.relative_to(BASE_DIR)))
            parts_imagens.append(part)

        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=20000),
        )
        modelos = listar_modelos_candidatos(client)

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=LaudoTecnicoMadeira,
            temperature=0.1,
        )
        conteudo = [
            *parts_imagens,
            "Realize a inspeção física desta peça de madeira, detecte e utilize a régua de 30cm para calibrar as dimensões reais, e preencha o laudo técnico estruturado.",
        ]

        ultimo_erro = None
        for modelo in modelos:
            try:
                response = client.models.generate_content(
                    model=modelo,
                    contents=conteudo,
                    config=config,
                )
                if response.parsed:
                    return RespostaBackend(
                        sucesso=True,
                        timestamp_iso=datetime.now().isoformat(timespec="seconds"),
                        arquivos_originais=originais_str,
                        arquivos_renomeados=renomeados_str,
                        modelo_utilizado=modelo,
                        tempo_processamento_segundos=round(time.time() - inicio, 2),
                        laudo=response.parsed,
                        erro=None,
                    )
            except (APIError, Exception) as err:
                ultimo_erro = err

        raise RuntimeError(f"Servidores indisponíveis no momento. Detalhe: {ultimo_erro}")

    except Exception as erro_fatal:
        return RespostaBackend(
            sucesso=False,
            timestamp_iso=datetime.now().isoformat(timespec="seconds"),
            arquivos_originais=originais_str,
            arquivos_renomeados=renomeados_str,
            modelo_utilizado=None,
            tempo_processamento_segundos=round(time.time() - inicio, 2),
            laudo=None,
            erro=str(erro_fatal),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Motor de Inspeção de Madeira com Calibração por Régua de 30cm e Saída JSON."
    )
    parser.add_argument(
        "imagens",
        nargs="*",
        help="Opcional: caminho da(s) foto(s). Se vazio, pega a foto mais recente da pasta.",
    )
    args = parser.parse_args()

    resultado = executar_inspecao(args.imagens)

    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(resultado.model_dump(), indent=2, ensure_ascii=False))

    if not resultado.sucesso:
        sys.exit(1)


if __name__ == "__main__":
    main()
