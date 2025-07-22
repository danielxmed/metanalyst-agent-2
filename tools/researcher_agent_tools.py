from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command
from langchain_core.messages import HumanMessage, ToolMessage
from typing import Annotated
import os


@tool
def literature_search(query: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """
    Realiza busca de literatura científica usando Tavily API com foco em periódicos médicos.
    
    Esta tool busca artigos científicos em bases de dados médicas confiáveis usando a 
    API Tavily. Retorna URLs de artigos relevantes formatados como texto.
    
    Args:
        query (str): Consulta de busca para literatura científica
        
    Returns:
        str: Resultado da busca formatado como texto
    """
    
    # Lista de domínios de periódicos e bases científicas em medicina
    medical_domains = [
        "pubmed.ncbi.nlm.nih.gov",
        "www.ncbi.nlm.nih.gov/pmc", 
        "www.cochranelibrary.com",
        "lilacs.bvsalud.org",
        "scielo.org",
        "www.embase.com",
        "www.webofscience.com",
        "www.scopus.com",
        "www.epistemonikos.org",
        "www.ebscohost.com",
        "www.tripdatabase.com",
        "pedro.org.au",
        "doaj.org",
        "scholar.google.com",
        "clinicaltrials.gov",
        "apps.who.int/trialsearch",
        "www.clinicaltrialsregister.eu",
        "www.isrctn.com",
        "www.thelancet.com",
        "www.nejm.org",
        "jamanetwork.com",
        "www.bmj.com",
        "www.nature.com/nm",
        "www.acpjournals.org/journal/aim",
        "journals.plos.org/plosmedicine",
        "www.jclinepi.com",
        "systematicreviewsjournal.biomedcentral.com",
        "ascopubs.org/journal/jco",
        "www.ahajournals.org/journal/circ",
        "www.gastrojournal.org",
        "academic.oup.com/eurheartj",
        "www.archives-pmr.org",
        "www.jacc.org",
        "www.scielo.br",
        "nejm.org",
        "thelancet.com",
        "bmj.com",
        "acpjournals.org/journal/aim",
        "cacancerjournal.com",
        "nature.com/nm",
        "cell.com/cell-metabolism/home",
        "thelancet.com/journals/langlo/home",
        "cochranelibrary.com",
        "memorias.ioc.fiocruz.br",
        "scielo.br/j/csp/",
        "cadernos.ensp.fiocruz.br",
        "scielo.br/j/rsp/",
        "scielo.org/journal/rpsp/",
        "journal.paho.org",
        "rbmt.org.br",
        "revistas.usp.br/rmrp",
        "ncbi.nlm.nih.gov/pmc",
        "scopus.com",
        "webofscience.com",
        "bvsalud.org",
        "jbi.global",
        "tripdatabase.com",
        "gov.br",
        "droracle.ai",
        "wolterskluwer.com",
        "semanticscholar.org",
        "globalindexmedicus.net",
        "sciencedirect.com"
    ]
    
    try:
        # Importa e configura o Tavily client
        from tavily import TavilyClient
        
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return Command(
                update={
                    "messages": [ToolMessage(
                        content="Erro: TAVILY_API_KEY não encontrada nas variáveis de ambiente.",
                        tool_call_id=tool_call_id
                    )]
                }
            )
        
        # Inicializa o cliente Tavily
        tavily_client = TavilyClient(api_key=api_key)
        
        # Realiza a busca com os parâmetros especificados
        response = tavily_client.search(
            query=query,
            search_depth="basic",
            max_results=10,
            include_domains=medical_domains
        )
        
        # Extrai as URLs dos resultados
        urls = []
        if "results" in response:
            for result in response["results"]:
                if "url" in result:
                    urls.append(result["url"])
        
        # Cria mensagem informativa sobre a busca realizada
        search_message = ToolMessage(
            content=f"🔍 Agente Researcher executou busca de literatura científica.\n"
                   f"Query: '{query}'\n" 
                   f"URLs encontradas: {len(urls)}\n"
                   f"Domínios pesquisados: periódicos e bases científicas médicas",
            tool_call_id=tool_call_id
        )
        
        # Retorna comando para atualizar o estado
        return Command(
            update={
                "urls_to_process": urls,  # Adiciona novos URLs sem remover os anteriores
                "previous_search_queries": [query],  # Adiciona query às buscas anteriores
                "messages": [search_message]  # Adiciona mensagem sobre a execução
            }
        )
        
    except ImportError:
        return Command(
            update={
                "messages": [ToolMessage(
                    content="Erro: Biblioteca tavily-python não instalada. Execute: pip install tavily-python",
                    tool_call_id=tool_call_id
                )]
            }
        )
    except Exception as e:
        return Command(
            update={
                "messages": [ToolMessage(
                    content=f"Erro na busca de literatura: {str(e)}",
                    tool_call_id=tool_call_id
                )]
            }
        )
