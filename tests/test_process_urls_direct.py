import os
from dotenv import load_dotenv
from tools.process_urls import process_urls
from state.state import MetaAnalysisState

# Carrega as variáveis de ambiente
load_dotenv()

print("=== TESTE DIRETO DA FERRAMENTA PROCESS_URLS ===")

# Cria um estado mock com algumas URLs para testar
test_state = MetaAnalysisState(
    user_request="Test request",
    urls_to_process=[
        "https://pubmed.ncbi.nlm.nih.gov/12466506/",
        "https://www.bmj.com/content/373/bmj.n991"
    ],
    processed_urls=[],
    messages=[],
    current_iteration=1,
    remaining_steps=5,
    meta_analysis_pico={"population": "test", "intervention": "test", "comparison": "test", "outcome": "test"},
    previous_search_queries=[],
    previous_retrieve_queries=[],
    retrieved_chunks=[],
    analysis_results=[],
    current_draft="",
    current_draft_iteration=1,
    reviewer_feedbacks=[],
    final_draft=""
)

print(f"URLs para processar: {len(test_state['urls_to_process'])}")
print(f"URLs processadas: {len(test_state['processed_urls'])}")

try:
    print("\n=== EXECUTANDO PROCESS_URLS ===")
    # Usar invoke em vez do método deprecated
    result = process_urls.invoke({
        "tool_call_id": "test_tool_call_id",
        "state": test_state
    })
    print(f"Resultado: {result}")
    
except Exception as e:
    print(f"ERRO: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n=== TESTE CONCLUÍDO ===")
