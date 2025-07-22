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

def test_accumulative_state():
    print("🧪 Testando se o estado é ACUMULATIVO (não substitui, mas adiciona)...")
    
    # Estado inicial com alguns dados já existentes
    initial_state = {
        "messages": [{"role": "user", "content": "Primeira busca"}],
        "metanalysis_pico": {
            "population": "pessoas com diabetes tipo 2",
            "intervention": "metformina", 
            "comparison": "placebo",
            "outcome": "controle de glicemia"
        },
        "user_request": "Testar estado acumulativo",
        "previous_search_queries": ["busca_anterior_1", "busca_anterior_2"],  # JÁ tem 2 queries
        "urls_to_process": ["url_existente_1.com", "url_existente_2.com"],    # JÁ tem 2 URLs
        "remaining_steps": 10,
        "current_iteration": 1,
        "previous_retrieve_queries": [],
        "processed_urls": [],
        "retrieved_chunks": [],
        "analysis_results": [],
        "current_draft_iteration": 0,
        "reviewer_feedbacks": []
    }
    
    print(f"📊 Estado INICIAL:")
    print(f"   - URLs já existentes: {len(initial_state['urls_to_process'])} -> {initial_state['urls_to_process']}")
    print(f"   - Queries já existentes: {len(initial_state['previous_search_queries'])} -> {initial_state['previous_search_queries']}")
    
    # Executa o agente com dados pré-existentes
    result = researcher_agent.invoke(initial_state, config={"recursion_limit": 15})
    
    print(f"\n📊 Estado APÓS execução:")
    print(f"   - Total URLs: {len(result.get('urls_to_process', []))}")
    print(f"   - Total Queries: {len(result.get('previous_search_queries', []))}")
    
    # Verifica se os dados antigos ainda existem
    old_urls_preserved = all(url in result.get('urls_to_process', []) 
                           for url in initial_state['urls_to_process'])
    old_queries_preserved = all(query in result.get('previous_search_queries', []) 
                              for query in initial_state['previous_search_queries'])
    
    print(f"\n✅ Verificações de ACUMULAÇÃO:")
    print(f"   - URLs antigos preservados: {'✅ SIM' if old_urls_preserved else '❌ NÃO'}")
    print(f"   - Queries antigas preservadas: {'✅ SIM' if old_queries_preserved else '❌ NÃO'}")
    
    # Lista os URLs antigos vs novos
    if old_urls_preserved:
        new_urls = [url for url in result.get('urls_to_process', []) 
                   if url not in initial_state['urls_to_process']]
        print(f"   - URLs NOVOS encontrados: {len(new_urls)}")
        if new_urls:
            print(f"     → Exemplos: {new_urls[:3]}...")
    
    # Lista as queries antigas vs novas  
    if old_queries_preserved:
        new_queries = [q for q in result.get('previous_search_queries', []) 
                      if q not in initial_state['previous_search_queries']]
        print(f"   - Queries NOVAS executadas: {len(new_queries)}")
        if new_queries:
            print(f"     → Novas queries: {new_queries}")
    
    if old_urls_preserved and old_queries_preserved:
        print(f"\n🎉 CONFIRMADO: O estado é ACUMULATIVO!")
        print(f"   ✓ Dados antigos são preservados")
        print(f"   ✓ Novos dados são adicionados (não substituídos)")
        print(f"   ✓ operator.add está funcionando corretamente")
    else:
        print(f"\n❌ PROBLEMA: Estado está sendo substituído, não acumulado!")
        
    return result

if __name__ == "__main__":
    test_accumulative_state()
