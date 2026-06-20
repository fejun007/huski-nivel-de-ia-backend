import os
import uuid
import struct
import json
import re
import requests
from fastapi import FastAPI, HTTPException, Header, Depends
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

SUPABASE_URL    = os.getenv("SUPABASE_URL", "https://jqvrmslrqpxesiuiyzuw.supabase.co")
SUPABASE_KEY    = os.getenv("SUPABASE_KEY", "")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
PARTNER_API_KEY = os.getenv("PARTNER_API_KEY", "")
FRONTEND_URL    = os.getenv("FRONTEND_URL", "https://huski-nivel-de-ia.vercel.app")

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


def require_partner_api_key(x_api_key: Optional[str] = Header(None)):
    if not PARTNER_API_KEY:
        raise HTTPException(status_code=503, detail="API de parceiros não configurada. Defina PARTNER_API_KEY.")
    if not x_api_key or x_api_key != PARTNER_API_KEY:
        raise HTTPException(status_code=401, detail="X-API-Key inválida ou ausente.")


def save_evaluation_result(session_id: Optional[str], nome: str, answers: list, result: dict):
    if not SUPABASE_KEY:
        return
    try:
        requests.post(
            f"{SUPABASE_URL}/rest/v1/avaliacoes_resultado",
            headers=supabase_headers(),
            json={
                "session_id": session_id,
                "nome": nome,
                "nivel": result.get("nivel"),
                "titulo": result.get("titulo"),
                "resumo": result.get("resumo"),
                "questoes": result.get("questoes"),
                "proximos_passos": result.get("proximos_passos"),
                "respostas": answers,
            },
            timeout=10,
        )
        if session_id:
            requests.patch(
                f"{SUPABASE_URL}/rest/v1/sessoes_avaliacao?id=eq.{session_id}",
                headers=supabase_headers(),
                json={"status": "concluido"},
                timeout=10,
            )
    except Exception as e:
        print(f"Erro ao salvar resultado no Supabase: {e}")


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

Para cada questão de múltipla escolha (tipo "mc"), as perguntas têm dificuldade (fácil/média/difícil) e uma resposta
tecnicamente correta já verificada objetivamente pelo sistema (indicado em "Resposta correta/incorreta" no contexto).
Escreva um "gabarito": confirme se a resposta do candidato estava correta ou incorreta, cite a alternativa certa
quando o candidato errar, e explique brevemente o conceito por trás da resposta correta.

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
    is_correct: Optional[bool] = None
    difficulty: Optional[str] = None   # 'facil' | 'media' | 'dificil'

class EvaluatePayload(BaseModel):
    nome: str
    answers: List[AnswerItem]
    session_id: Optional[str] = None


class ExternalSessionCreatePayload(BaseModel):
    nome: str
    external_id: Optional[str] = None


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
        if a.difficulty:
            lines.append(f"Dificuldade da questão: {a.difficulty}")
        resp = f"{a.option_label} — {a.answer_text}" if a.option_label else a.answer_text
        lines.append(f"Resposta: {resp}")
        if a.is_correct is not None:
            lines.append(f"Resposta {'correta' if a.is_correct else 'incorreta'} (verificado objetivamente pelo sistema).")
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

    save_evaluation_result(
        session_id=payload.session_id,
        nome=payload.nome,
        answers=[a.model_dump() for a in payload.answers],
        result=result,
    )

    return result


# ── API para integração com outras aplicações ──────────────────────

@app.post("/api/v1/avaliacoes")
def api_create_avaliacao(payload: ExternalSessionCreatePayload, _auth=Depends(require_partner_api_key)):
    if not SUPABASE_KEY:
        raise HTTPException(status_code=503, detail="Supabase não configurado no servidor.")
    try:
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/sessoes_avaliacao",
            headers=supabase_headers(),
            json={
                "nome": payload.nome,
                "status": "em_andamento",
                "external_id": payload.external_id,
            },
            timeout=10,
        )
        if not r.ok:
            raise HTTPException(status_code=502, detail=f"Erro Supabase: {r.status_code} {r.text}")
        session_id = r.json()[0]["id"]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    assessment_url = f"{FRONTEND_URL}/?session_id={session_id}&nome={payload.nome}"
    return {"session_id": session_id, "assessment_url": assessment_url}


@app.get("/api/v1/avaliacoes/{session_id}")
def api_get_avaliacao(session_id: str, _auth=Depends(require_partner_api_key)):
    if not SUPABASE_KEY:
        raise HTTPException(status_code=503, detail="Supabase não configurado no servidor.")
    try:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/avaliacoes_resultado",
            headers=supabase_headers(),
            params={"session_id": f"eq.{session_id}", "order": "created_at.desc", "limit": 1},
            timeout=10,
        )
        if not r.ok:
            raise HTTPException(status_code=502, detail=f"Erro Supabase: {r.status_code} {r.text}")
        rows = r.json()
        if rows:
            row = rows[0]
            return {
                "session_id": session_id,
                "status": "concluido",
                "nome": row.get("nome"),
                "nivel": row.get("nivel"),
                "titulo": row.get("titulo"),
                "resumo": row.get("resumo"),
                "questoes": row.get("questoes"),
                "proximos_passos": row.get("proximos_passos"),
                "criado_em": row.get("created_at"),
            }

        r2 = requests.get(
            f"{SUPABASE_URL}/rest/v1/sessoes_avaliacao",
            headers=supabase_headers(),
            params={"id": f"eq.{session_id}", "limit": 1},
            timeout=10,
        )
        if r2.ok and r2.json():
            return {"session_id": session_id, "status": r2.json()[0].get("status", "em_andamento")}

        raise HTTPException(status_code=404, detail="Sessão não encontrada.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
