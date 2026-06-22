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


def save_evaluation_result(session_id: Optional[str], nome: str, answers: list, result: dict, email: Optional[str] = None, cpf: Optional[str] = None, pontuacao: Optional[dict] = None):
    if not SUPABASE_KEY:
        return
    try:
        requests.post(
            f"{SUPABASE_URL}/rest/v1/avaliacoes_resultado",
            headers=supabase_headers(),
            json={
                "session_id": session_id,
                "nome": nome,
                "email": email,
                "cpf": cpf,
                "nivel": result.get("nivel"),
                "titulo": result.get("titulo"),
                "resumo": result.get("resumo"),
                "questoes": result.get("questoes"),
                "proximos_passos": result.get("proximos_passos"),
                "pontuacao": pontuacao,
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

NIVEL_TITULOS = {
    1: "Usuário Casual / Explorador",
    2: "Usuário Funcional / Operador de Prompts",
    3: "Integrador / Otimizador de Processos",
    4: "Arquiteto de Soluções / Desenvolvedor de IA",
    5: "Estrategista / Cientista de IA",
}

DIFFICULTY_WEIGHT = {"facil": 1, "media": 2, "dificil": 3}

# Pesos dos três blocos do teste na nota final (somam 1.0)
WEIGHT_SELF_EVAL    = 0.20  # 5 perguntas de autoavaliação (diagnóstico, sem certo/errado)
WEIGHT_TECH_ADAPT   = 0.50  # 6 perguntas técnicas adaptativas (objetivas, ponderadas por dificuldade)
WEIGHT_CASE_REVIEW  = 0.30  # 2 cases + 2 reviews (nota_nivel do Gemini por critérios da rubrica)


def compute_quantitative_level(answers: List["AnswerItem"], questoes: list) -> tuple:
    """Calcula o nível final (1-5) combinando os três blocos do teste com pesos fixos.
    Retorna (nivel_final:int, detalhes:dict) para auditoria/transparência."""

    # 1) Autoavaliação: média do nível (1-5) representado pela opção escolhida
    self_levels = [a.self_level for a in answers if a.question_type == "mc" and a.self_level is not None]
    self_eval_score = sum(self_levels) / len(self_levels) if self_levels else None

    # 2) Técnicas adaptativas: % de pontos conquistados (ponderado por dificuldade) -> escala 1-5
    earned = 0
    possible = 0
    for a in answers:
        if a.question_type == "mc" and a.difficulty in DIFFICULTY_WEIGHT and a.is_correct is not None:
            w = DIFFICULTY_WEIGHT[a.difficulty]
            possible += w
            if a.is_correct:
                earned += w
    tech_score = (1 + (earned / possible) * 4) if possible > 0 else None

    # 3) Cases/reviews: média das nota_nivel atribuídas pelo Gemini
    notas = []
    for q in questoes or []:
        if q.get("tipo") in ("case", "review") and q.get("nota_nivel") is not None:
            try:
                notas.append(float(q["nota_nivel"]))
            except (TypeError, ValueError):
                pass
    case_review_score = sum(notas) / len(notas) if notas else None

    # Combina os blocos disponíveis, renormalizando os pesos se algum bloco faltar
    components = [
        (self_eval_score, WEIGHT_SELF_EVAL),
        (tech_score, WEIGHT_TECH_ADAPT),
        (case_review_score, WEIGHT_CASE_REVIEW),
    ]
    available = [(score, w) for score, w in components if score is not None]
    if not available:
        final_score = 3.0
    else:
        total_w = sum(w for _, w in available)
        final_score = sum(score * w for score, w in available) / total_w

    nivel_final = max(1, min(5, round(final_score)))

    return nivel_final, {
        "self_eval_score": round(self_eval_score, 2) if self_eval_score is not None else None,
        "tech_score": round(tech_score, 2) if tech_score is not None else None,
        "case_review_score": round(case_review_score, 2) if case_review_score is not None else None,
        "final_score": round(final_score, 2),
        "tech_pontos": f"{earned}/{possible}" if possible else None,
    }


EVAL_SYSTEM = f"""Você é ARIA, avaliadora sênior de maturidade em IA da Huski.
Avalie o usuário com base nas respostas do teste estruturado e retorne SOMENTE JSON válido.

RUBRICA:
{RUBRICA}

Para cada questão de múltipla escolha (tipo "mc") existem dois grupos:
1) Perguntas de autoavaliação (sem indicação de "Resposta correta/incorreta" no contexto) — são diagnósticas, não
   existe certo ou errado. Escreva um "gabarito" explicando o que aquela resposta específica revela sobre o nível
   de maturidade em IA do candidato.
2) Perguntas técnicas adaptativas (com dificuldade fácil/média/difícil e "Resposta correta/incorreta" já verificada
   objetivamente pelo sistema no contexto). Escreva um "gabarito" confirmando se a resposta estava correta ou
   incorreta, citando a alternativa certa quando o candidato errar, e explicando brevemente o conceito por trás
   da resposta correta. Leve em conta a progressão de dificuldade (acertar perguntas difíceis pesa mais do que
   acertar perguntas fáceis) ao definir o nível final do candidato.

Para cada case e review reverso (tipo "case"/"review"), avalie a resposta do usuário contra os critérios por nível
fornecidos na questão e escreva uma "critica" (2-4 frases, específica à resposta dada, citando o que faltou ou o que
foi bem feito) e uma "nota_nivel" (1-5, pode usar números inteiros) que reflete em que nível da rubrica aquela
resposta específica se encaixa. A "nota_nivel" é o dado mais importante desta avaliação — será usada por um cálculo
quantitativo externo para definir o nível final do candidato, então seja rigoroso e consistente com a rubrica.

O campo "nivel"/"titulo" que você retornar é apenas um rascunho e será recalculado por uma fórmula quantitativa
externa (baseada em acertos ponderados por dificuldade nas perguntas técnicas e na média das "nota_nivel" de
cases/reviews) — não é o valor final exibido ao candidato. Por isso, no "resumo", descreva o perfil e o comportamento
observado nas respostas sem afirmar categoricamente "você está no nível X", pois o número final pode ser ajustado.

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
    email: Optional[str] = None
    cpf: Optional[str] = None

class AnswerItem(BaseModel):
    question_id: str
    question_type: str      # 'mc' | 'case' | 'review'
    question_text: str
    answer_text: str
    option_label: Optional[str] = None
    artifact: Optional[str] = None     # prompt/fluxo a ser analisado em questões case/review
    criteria: Optional[str] = None
    is_correct: Optional[bool] = None
    difficulty: Optional[str] = None   # 'facil' | 'media' | 'dificil'
    self_level: Optional[int] = None   # nível (1-5) representado pela opção escolhida em perguntas de autoavaliação

class EvaluatePayload(BaseModel):
    nome: str
    answers: List[AnswerItem]
    session_id: Optional[str] = None
    email: Optional[str] = None
    cpf: Optional[str] = None


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
            json={
                "nome": payload.nome,
                "resposta_inicial": payload.resposta_inicial,
                "status": "em_andamento",
                "email": payload.email,
                "cpf": payload.cpf,
            },
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
        if a.artifact:
            lines.append(f"Artefato analisado pelo candidato (prompt/fluxo apresentado na questão):\n{a.artifact}")
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

    nivel_final, score_breakdown = compute_quantitative_level(payload.answers, result.get("questoes"))
    result["nivel"] = nivel_final
    result["titulo"] = NIVEL_TITULOS[nivel_final]
    result["pontuacao"] = score_breakdown

    save_evaluation_result(
        session_id=payload.session_id,
        nome=payload.nome,
        answers=[a.model_dump() for a in payload.answers],
        result=result,
        email=payload.email,
        cpf=payload.cpf,
        pontuacao=score_breakdown,
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


@app.get("/api/v1/avaliacoes")
def api_list_avaliacoes(email: Optional[str] = None, cpf: Optional[str] = None, _auth=Depends(require_partner_api_key)):
    if not SUPABASE_KEY:
        raise HTTPException(status_code=503, detail="Supabase não configurado no servidor.")
    if not email and not cpf:
        raise HTTPException(status_code=400, detail="Informe email ou cpf como query param.")
    try:
        params = {"order": "created_at.desc", "limit": 200}
        if email:
            params["email"] = f"eq.{email}"
        else:
            params["cpf"] = f"eq.{cpf}"
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/avaliacoes_resultado",
            headers=supabase_headers(),
            params=params,
            timeout=10,
        )
        if not r.ok:
            raise HTTPException(status_code=502, detail=f"Erro Supabase: {r.status_code} {r.text}")
        rows = r.json()
        resultados = [{
            "session_id": row.get("session_id"),
            "nome": row.get("nome"),
            "nivel": row.get("nivel"),
            "titulo": row.get("titulo"),
            "resumo": row.get("resumo"),
            "questoes": row.get("questoes"),
            "proximos_passos": row.get("proximos_passos"),
            "pontuacao": row.get("pontuacao"),
            "criado_em": row.get("created_at"),
        } for row in rows]
        return {"total": len(resultados), "resultados": resultados}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


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
