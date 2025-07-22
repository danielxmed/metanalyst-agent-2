#!/usr/bin/env python3
"""
Teste básico para verificar a implementação da tool literature_search
"""

import os
from tools.researcher_agent_tools import literature_search

def test_literature_search_tool():
    """Testa se a tool literature_search está funcionando corretamente"""
    
    print("🧪 Testando a implementação da tool literature_search...")
    
    # Teste 1: Verificar se a tool existe e pode ser importada
    print("✅ Tool literature_search importada com sucesso")
    
    # Teste 2: Verificar se a tool tem a assinatura correta
    assert hasattr(literature_search, 'name'), "Tool deve ter atributo 'name'"
    assert hasattr(literature_search, 'description'), "Tool deve ter atributo 'description'"
    print(f"✅ Nome da tool: {literature_search.name}")
    print(f"✅ Descrição da tool: {literature_search.description[:100]}...")
    
    # Teste 3: Verificar se consegue executar sem erro (mesmo sem API key)
    try:
        # Simula execução sem API key para testar tratamento de erro
        if 'TAVILY_API_KEY' in os.environ:
            del os.environ['TAVILY_API_KEY']
        
        result = literature_search.invoke({"query": "diabetes mellitus treatment"})
        print("✅ Tool executou sem erro (tratamento de erro funcionando)")
        print(f"✅ Tipo do resultado: {type(result)}")
        
        # Verifica se retorna um Command
        from langgraph.types import Command
        assert isinstance(result, Command), "Tool deve retornar um Command do LangGraph"
        print("✅ Tool retorna Command corretamente")
        
    except Exception as e:
        print(f"❌ Erro na execução da tool: {e}")
        return False
    
    print("\n🎉 Todos os testes passaram! A tool literature_search está implementada corretamente.")
    return True

if __name__ == "__main__":
    test_literature_search_tool()
