#!/usr/bin/env python3
import os
import sys
import importlib.util
from dotenv import load_dotenv

# Adiciona o diretório raiz do projeto ao Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv(os.path.join(project_root, '.env'))

# Importa o módulo researcher diretamente evitando o __init__.py
spec = importlib.util.spec_from_file_location("researcher", os.path.join(project_root, "agents", "researcher.py"))
researcher_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(researcher_module)

researcher_agent = researcher_module.researcher_agent


result = researcher_agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Buscar estudos sobre diabetes tipo 2 e metformina"
            }
        ],
        "metanalysis_pico": {
            "population": "pessoas com diabetes tipo 2",
            "intervention": "metformina",
            "comparison": "placebo",
            "outcome": "controle de glicemia"
        },
        "user_request": "Buscar estudos sobre diabetes tipo 2 e metformina",
        "previous_search_queries": [],
        "remaining_steps": 15,  # Limite de passos para evitar recursão infinita
        "current_iteration": 1,
        "urls_to_process": [],
        "previous_retrieve_queries": [],
        "processed_urls": [],
        "retrieved_chunks": [],
        "analysis_results": [],
        "current_draft_iteration": 0,
        "reviewer_feedbacks": []
    },
    config={"recursion_limit": 20}  # Aumenta o limite de recursão
)

print(result)
