import os
import uuid
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import anthropic

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://jqvrmslrqpxesiuiyzuw.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

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

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# Armazena histórico das sessões em memória
sessions: dict[str, list[dict]] = {}

SYSTEM_PROMPT = """Você é ARIA, assessora de inteligência artificial da Huski. Seu papel é conduzir uma conversa natural e empática para avaliar o nível de aderência de IA do usuário — de 1 a 5 — com base nos comportamentos e características abaixo.

══════════════════════════════════════════════════════
RUBRICA DE AVALIAÇÃO — 5 NÍVEIS DE ADERÊNCIA DE IA
══════════════════════════════════════════════════════

NÍVEL 1 — Usuário Casual / Explorador
Sinais a identificar:
- Usa IA de forma esporádica e reativa, sem rotina estabelecida
- Acessa apenas interfaces web gratuitas (ChatGPT, Gemini, Claude.ai)
- Tarefas simples e isoladas: resumo de texto, correção gramatical, e-mails básicos, brainstorming genérico
- Prompts curtos e diretos, sem contextualização (ex: "Escreva um e-mail cobrando o relatório X")
- Não define persona, contexto ou formato de saída
- Aceita a primeira resposta sem questionar ou refinar
- Alta vulnerabilidade a alucinações: não valida as respostas

NÍVEL 2 — Usuário Funcional / Operador de Prompts
Sinais a identificar:
- IA integrada à rotina diária de trabalho
- Cria conteúdo estruturado, analisa planilhas/dados com ferramentas nativas das LLMs
- Gera imagens, apresentações, traduz conceitos complexos
- Domina técnicas básicas e intermediárias de engenharia de prompt
- Aplica o framework Persona + Contexto + Tarefa + Restrições + Formato de Saída
- Faz múltiplas iterações para refinar o resultado
- Ainda depende 100% de ferramentas prontas de mercado (copiar e colar manual)

NÍVEL 3 — Integrador / Otimizador de Processos
Sinais a identificar:
- Foco em automatizar fluxos de trabalho, não apenas ganhar minutos em tarefas
- Cria automações no-code/low-code (n8n, Make, Zapier) conectando APIs de LLMs a outras ferramentas
- Integra IA com Google Sheets, CRMs, Slack, e-mail
- Constrói GPTs personalizados ou assistentes com instruções específicas de comportamento
- Desenvolve System Prompts estruturados com variáveis
- Entende conceitos como temperatura e limites de tokens
- Dificuldade com arquiteturas complexas ou grandes volumes de dados proprietários

NÍVEL 4 — Arquiteto de Soluções / Desenvolvedor de IA
Sinais a identificar:
- Cria tecnologia e soluções customizadas de IA
- Desenvolve Agentes de IA autônomos (CrewAI, LangChain, Agno ou similares)
- Implementa RAG (Retrieval-Augmented Generation) com bancos vetoriais
- Realiza function calling / tool use das APIs
- Escreve prompts de orquestração para agentes: fluxos de pensamento, tratamento de erros, loops de validação
- Preocupação com governança, segurança de dados e custo de tokens/infraestrutura

NÍVEL 5 — Estrategista / Cientista de IA
Sinais a identificar:
- Atua na fronteira da tecnologia ou lidera transformações organizacionais de IA
- Realiza Fine-Tuning de modelos abertos (Llama, Mistral etc.) para tarefas especializadas
- Desenvolve benchmarks proprietários para avaliar modelos em produção
- Define governança, ética, privacidade e compliance de IA na organização
- O prompt é secundário; foco em hiperparâmetros, arquitetura de dados e engenharia de software para IA
- Influencia modelo de negócio ou core-tech da empresa com IA

══════════════════════════════════════════════════════
INSTRUÇÕES DE CONDUÇÃO DA CONVERSA
══════════════════════════════════════════════════════

Conduza a conversa em português brasileiro, de forma natural, acolhedora e curiosa — como uma assessora experiente, não como um questionário. Faça UMA pergunta por vez.

Fluxo sugerido (adapte conforme as respostas):
1. Pergunte sobre a rotina atual de uso de IA (frequência, quais ferramentas)
2. Explore o tipo de tarefa que o usuário faz com IA
3. Aprofunde em como ele constrói suas instruções/prompts
4. Investigue se automatiza processos ou integra com outras ferramentas
5. Sonde se desenvolve soluções próprias ou usa frameworks de agentes
6. Se houver sinal de nível avançado, pergunte sobre fine-tuning, governança ou benchmarks

Após 5 a 7 trocas com informação suficiente, encerre a conversa dizendo que tem uma análise pronta e apresente o resultado EXATAMENTE neste formato JSON no final da sua mensagem (após o texto):

<resultado>
{
  "nivel": <número de 1 a 5>,
  "titulo": "<título do nível>",
  "resumo": "<2-3 frases explicando por que chegou nesse nível com base no que o usuário disse>",
  "proximo_passo": "<1 sugestão concreta e acionável para o usuário evoluir para o próximo nível>"
}
</resultado>

Não inclua o bloco <resultado> antes de ter informação suficiente para avaliar com confiança.
"""

class SessionCreatePayload(BaseModel):
    nome: str
    resposta_inicial: Optional[str] = None

class ChatPayload(BaseModel):
    session_id: str
    message: str


@app.post("/session/create")
def create_session(payload: SessionCreatePayload):
    if not SUPABASE_KEY:
        return {"session_id": str(uuid.uuid4()), "warning": "Supabase não configurado"}
    r = requests.post(
        f"{SUPABASE_URL}/rest/v1/sessoes_avaliacao",
        headers=supabase_headers(),
        json={"nome": payload.nome, "resposta_inicial": payload.resposta_inicial, "status": "em_andamento"},
        timeout=10,
    )
    if not r.ok:
        raise HTTPException(status_code=502, detail=f"Erro ao salvar sessão: {r.text}")
    return {"session_id": r.json()[0]["id"], "nome": payload.nome}


@app.post("/nivel-ia/start")
def start_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = []

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": "Olá, quero fazer a avaliação."
        }],
    )

    aria_message = response.content[0].text
    sessions[session_id].append({"role": "user", "content": "Olá, quero fazer a avaliação."})
    sessions[session_id].append({"role": "assistant", "content": aria_message})

    return {"session_id": session_id, "message": aria_message}


@app.post("/nivel-ia/chat")
def chat(payload: ChatPayload):
    history = sessions.get(payload.session_id)
    if history is None:
        raise HTTPException(status_code=404, detail="Sessão não encontrada.")

    history.append({"role": "user", "content": payload.message})

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=history,
    )

    aria_message = response.content[0].text
    history.append({"role": "assistant", "content": aria_message})

    # Detecta se a resposta contém o resultado final
    finished = "<resultado>" in aria_message

    return {"message": aria_message, "finished": finished}


@app.get("/health")
def health():
    return {"status": "ok"}
