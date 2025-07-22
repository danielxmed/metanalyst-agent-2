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

# Teste para verificar se o estado está sendo atualizado
def test_state_update():
    print("🧪 Testando se o estado é atualizado pelas tools...")
    
    # Estado inicial
    initial_state = {
        "messages": [
            {
                "role": "user", 
                "content": "Testar busca de diabetes"
            }
        ],
        "metanalysis_pico": {
            "population": "pessoas com diabetes tipo 2",
            "intervention": "metformina", 
            "comparison": "placebo",
            "outcome": "controle de glicemia"
        },
        "user_request": "Testar busca de diabetes",
        "previous_search_queries": [],
        "urls_to_process": []  # Campo vazio inicialmente
    }
    
    print(f"📊 Estado inicial:")
    print(f"   - URLs para processar: {len(initial_state.get('urls_to_process', []))}")
    print(f"   - Queries anteriores: {len(initial_state.get('previous_search_queries', []))}")
    print(f"   - Mensagens: {len(initial_state.get('messages', []))}")
    
    # Executa o agente
    result = researcher_agent.invoke(initial_state)
    
    print(f"\n📊 Estado após execução:")
    print(f"   - URLs para processar: {len(result.get('urls_to_process', []))}")
    print(f"   - Queries anteriores: {len(result.get('previous_search_queries', []))}")
    print(f"   - Mensagens: {len(result.get('messages', []))}")
    
    # Verifica se o estado foi atualizado
    urls_updated = len(result.get('urls_to_process', [])) > len(initial_state.get('urls_to_process', []))
    queries_updated = len(result.get('previous_search_queries', [])) > len(initial_state.get('previous_search_queries', []))
    messages_updated = len(result.get('messages', [])) > len(initial_state.get('messages', []))
    
    print(f"\n✅ Verificações:")
    print(f"   - URLs foram adicionados: {'✅ SIM' if urls_updated else '❌ NÃO'}")
    print(f"   - Queries foram adicionadas: {'✅ SIM' if queries_updated else '❌ NÃO'}")
    print(f"   - Mensagens foram adicionadas: {'✅ SIM' if messages_updated else '❌ NÃO'}")
    
    if urls_updated and queries_updated and messages_updated:
        print(f"\n🎉 SUCESSO: O estado foi atualizado corretamente pelas tools!")
        print(f"   - {len(result.get('urls_to_process', []))} URLs encontrados")
        print(f"   - {len(result.get('previous_search_queries', []))} queries executadas")
    else:
        print(f"\n❌ FALHA: O estado não foi atualizado como esperado")
        
    return result

if __name__ == "__main__":
    test_state_update()
