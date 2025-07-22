# 🤖 Metanalyst Agent

Um sistema de agentes inteligentes para conduzir metanálises automatizadas de literatura científica.

## 📋 Visão Geral

O Metanalyst Agent é um sistema multi-agente que automatiza o processo de condução de metanálises, desde a pesquisa de literatura até a geração do documento final. O sistema utiliza agentes especializados para diferentes etapas do processo.

## ⚙️ Configuração

### 1. Pré-requisitos

- Python 3.8+
- Chaves de API para:
  - OpenAI (para GPT)
  - Anthropic (para Claude)
  - Tavily (para pesquisa na web)

### 2. Instalação

```bash
# Clone o repositório
git clone https://github.com/danielxmed/metanalyst-agent-2.git
cd metanalyst-agent-2

# Instale as dependências
pip install -r requirements.txt
```

### 3. Configuração das Variáveis de Ambiente

1. Copie o arquivo de exemplo:
```bash
cp .env.example .env
```

2. Edite o arquivo `.env` e adicione suas chaves de API:
```env
OPENAI_API_KEY=sua_chave_openai_aqui
ANTHROPIC_API_KEY=sua_chave_anthropic_aqui
TAVILY_API_KEY=sua_chave_tavily_aqui
```

**⚠️ IMPORTANTE: Nunca commit suas chaves de API reais! O arquivo `.env` está no `.gitignore` para proteger suas credenciais.**

## 🏗️ Arquitetura

O sistema é composto por:

- **Supervisor Agent**: Coordena o fluxo de trabalho
- **Researcher Agent**: Realiza pesquisas de literatura
- **State Manager**: Mantém o estado do processo

## 📊 Estado Atual do Processo

| Chave do Estado                | Descrição                                      |
|------------------------------- |------------------------------------------------|
| **current_iteration**          | Iteração atual do fluxo                        |
| **messages**                   | Mensagens trocadas até o momento               |
| **metanalysis_pico**           | Elementos PICO definidos                       |
| **user_request**               | Solicitação original do usuário                |
| **previous_search_queries**     | Pesquisas anteriores realizadas                |
| **urls_to_process**            | URLs a serem processadas                       |
| **processed_urls**             | URLs já processadas                            |
| **retrieved_chunks**           | Trechos recuperados do repositório             |
| **previous_retrieve_queries**   | Consultas de recuperação anteriores            |
| **analysis_results**           | Resultados das análises                        |
| **current_draft**              | Rascunho atual da metanálise                   |
| **current_draft_iteration**    | Iteração do rascunho atual                     |
| **reviewer_feedbacks**         | Feedbacks do revisor                           |
| **final_draft**                | Versão final da metanálise                     |

> _Cada linha representa uma chave do estado mantido durante a execução do pipeline de metanálise._

## 🚀 Como Usar

```python
# Exemplo de uso básico
from agents.supervisor import SupervisorAgent
from state.state import State

# Inicializar o estado
state = State()

# Criar o agente supervisor
supervisor = SupervisorAgent()

# Executar uma metanálise
result = supervisor.run("Metanálise sobre eficácia de intervenções em saúde mental")
```

## 📁 Estrutura do Projeto

```
metanalyst-agent-2/
├── agents/                 # Agentes do sistema
│   ├── supervisor.py      # Agente supervisor
│   └── researcher.py      # Agente pesquisador
├── prompts/               # Templates de prompts
├── state/                 # Gerenciamento de estado
├── tools/                 # Ferramentas dos agentes
├── tests/                 # Testes automatizados
├── .env.example          # Exemplo de configuração
└── requirements.txt      # Dependências Python
```

## 🧪 Testes

Execute os testes para verificar se tudo está funcionando:

```bash
python -m pytest tests/
```

## 🛡️ Segurança

- Nunca committe arquivos `.env` com chaves reais
- Use variáveis de ambiente para configurações sensíveis
- Mantenha suas chaves de API seguras e não as compartilhe

## 📄 Licença

Este projeto está licenciado sob a licença MIT.

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor, abra uma issue ou pull request.
