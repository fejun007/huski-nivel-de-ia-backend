import os
import uuid
import struct
import json
import re
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional, List

try:
    from google import genai
    from google.genai import types as genai_types
    _genai_ok = True
except ImportError:
    _genai_ok = False

SUPABASE_URL   = os.getenv("SUPABASE_URL", "https://jqvrmslrqpxesiuiyzuw.supabase.co")
SUPABASE_KEY   = os.getenv("SUPABASE_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if (_genai_ok and GEMINI_API_KEY) else None


def _pcm_to_wav(pcm: bytes, rate: int = 24000, channels: int = 1, bits: int = 16) -> bytes:
    data_size = len(pcm)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", data_size + 36, b"WAVE",
        b"fmt ", 16, 1, channels, rate,
        rate * channels * bits // 8,
        channels * bits // 8, bits,
        b"data", data_size,
    )
    return header + pcm


def supabase_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


app = FastAPI(title="Huski — Nível de IA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RUBRICA = """
NÍVEL 1 — Usuário Casual / Explorador
- Uso esporádico e reativo, sem rotina
- Apenas interfaces web gratuitas (ChatGPT, Gemini, Claude.ai)
- Tarefas simples e isoladas; prompts curtos sem estrutura
- Alta vulnerabilidade a alucinações; não valida respostas

NÍVEL 2 — Usuário Funcional / Operador de Prompts
- IA integrada à rotina diária
- Cria conteúdo estruturado, analisa dados com ferramentas nativas
- Domina Persona + Contexto + Tarefa + Restrições + Formato de Saída
- Faz múltiplas iterações; ainda depende 100% de ferramentas prontas

NÍVEL 3 — Integrador / Otimizador de Processos
- Foco em automatizar fluxos de trabalho inteiros
- Cria automações no-code/low-code (n8n, Make, Zapier) conectando APIs
- Constrói GPTs personalizados ou assistentes com System Prompts estruturados
- Entende temperatura, limites de tokens, variáveis

NÍVEL 4 — Arquiteto de Soluções / Desenvolvedor de IA
- Desenvolve Agentes de IA autônomos (CrewAI, LangChain, Agno)
- Implementa RAG com bancos vetoriais; usa function calling / tool use
- Escreve prompts de orquestração; preocupação com governança e custo

NÍVEL 5 — Estrategista / Cientista de IA
- Fine-tuning de modelos abertos (Llama, Mistral)
- Desenvolve benchmarks proprietários em produção
- Define governança, ética, privacidade e compliance de IA
- Influencia modelo de negócio ou core-tech da empresa
"""

EVAL_SYSTEM = f"""Você é ARIA, avaliadora sênior de maturidade em IA da Huski.
Avalie o usuário com base nas respostas do teste estruturado e retorne SOMENTE JSON válido.

RUBRICA:
{RUBRICA}

Para cada questão de múltipla escolha (tipo "mc"), escreva um "gabarito": uma frase curta explicando o que aquela
resposta específica revela sobre o nível de maturidade em IA do candidato (não existe "certo ou errado", é diagnóstico).

Para cada case e review reverso (tipo "case"/"review"), avalie a resposta do usuário contra os critérios por nível
fornecidos na questão e escreva uma "critica" (2-4 frases, específica à resposta dada, citando o que faltou ou o que
foi bem feito) e uma "nota_nivel" (1-5) que reflete em que nível da rubrica aquela resposta específica se encaixa.

Formato obrigatório (sem texto fora do JSON):
{{
  "nivel": <1-5>,
  "titulo": "<título exato do nível>",
  "resumo": "<2-3 frases referenciando respostas específicas do teste>",
  "questoes": [
    {{
      "question_id": "<id da questão>",
      "tipo": "mc" | "case" | "review",
      "pergunta": "<texto da questão>",
      "resposta_usuario": "<resposta dada pelo usuário>",
      "gabarito": "<obrigatório apenas para tipo mc>",
      "critica": "<obrigatório apenas para tipo case/review>",
      "nota_nivel": "<obrigatório apenas para tipo case/review, 1-5>"
    }}
  ],
  "proximos_passos": [
    "<ação concreta 1 para evoluir de nível>",
    "<ação concreta 2 para evoluir de nível>",
    "<ação concreta 3 para evoluir de nível>"
  ]
}}"""


class TTSPayload(BaseModel):
    text: str

class SessionCreatePayload(BaseModel):
    nome: str
    resposta_inicial: Optional[str] = None

class AnswerItem(BaseModel):
    question_id: str
    question_type: str      # 'mc' | 'case' | 'review'
    question_text: str
    answer_text: str
    option_label: Optional[str] = None
    criteria: Optional[str] = None

class EvaluatePayload(BaseModel):
    nome: str
    answers: List[AnswerItem]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "tts": gemini_client is not None,
        "llm": gemini_client is not None,
    }


@app.post("/tts")
def text_to_speech(payload: TTSPayload):
    if not gemini_client:
        raise HTTPException(status_code=503, detail="TTS não configurado. Defina GEMINI_API_KEY.")
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=payload.text,
            config=genai_types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=genai_types.SpeechConfig(
                    voice_config=genai_types.VoiceConfig(
                        prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(voice_name="Zephyr")
                    )
                ),
            ),
        )
        part = response.candidates[0].content.parts[0]
        audio_bytes = part.inline_data.data
        mime = part.inline_data.mime_type or "audio/pcm"
        if "pcm" in mime.lower() or "raw" in mime.lower() or mime == "audio/l16":
            rate = 24000
            if "rate=" in mime:
                try:
                    rate = int(mime.split("rate=")[1].split(";")[0].strip())
                except Exception:
                    pass
            audio_bytes = _pcm_to_wav(audio_bytes, rate)
            mime = "audio/wav"
        return Response(content=audio_bytes, media_type=mime, headers={"Cache-Control": "no-store"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro TTS: {str(e)}")


@app.post("/session/create")
def create_session(payload: SessionCreatePayload):
    if not SUPABASE_KEY:
        return {"session_id": str(uuid.uuid4()), "warning": "Supabase não configurado"}
    try:
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/sessoes_avaliacao",
            headers=supabase_headers(),
            json={"nome": payload.nome, "resposta_inicial": payload.resposta_inicial, "status": "em_andamento"},
            timeout=10,
        )
        if not r.ok:
            return {"session_id": str(uuid.uuid4()), "warning": f"Supabase error: {r.status_code}"}
        return {"session_id": r.json()[0]["id"], "nome": payload.nome}
    except Exception as e:
        return {"session_id": str(uuid.uuid4()), "warning": str(e)}


@app.post("/nivel-ia/evaluate")
def evaluate(payload: EvaluatePayload):
    if not gemini_client:
        raise HTTPException(status_code=503, detail="Gemini não configurado. Defina GEMINI_API_KEY.")

    lines = [f"Candidato: {payload.nome}\n"]
    for i, a in enumerate(payload.answers, 1):
        tipo = {"mc": "Múltipla Escolha", "case": "Case", "review": "Review Reverso"}.get(a.question_type, a.question_type)
        lines.append(f"[{i}] {tipo} — {a.question_text}")
        resp = f"{a.option_label} — {a.answer_text}" if a.option_label else a.answer_text
        lines.append(f"Resposta: {resp}")
        if a.criteria:
            lines.append(f"Critérios de avaliação desta questão: {a.criteria}")
        lines.append("")

    prompt = EVAL_SYSTEM + "\n\n" + "\n".join(lines)

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        raw = response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro Gemini: {str(e)}")

    try:
        result = json.loads(raw)
    except Exception:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            result = json.loads(m.group())
        else:
            raise HTTPException(status_code=500, detail="Erro ao parsear avaliação do modelo.")

    return result
