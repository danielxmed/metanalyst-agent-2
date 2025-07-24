#!/usr/bin/env python3
"""
Interactive CLI for Meta-Analysis
=================================

This CLI allows you to test different queries for the meta-analysis system
and display the results in a stream in an organized and aesthetically pleasing way.
"""

import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, Any, List
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich.align import Align
from rich import box
from rich.markdown import Markdown
from rich.syntax import Syntax

# Import the supervisor agent
from agents.supervisor import supervisor_agent
from state.state import MetaAnalysisState
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MetaAnalysisCLI:
    def __init__(self):
        self.console = Console()
        self.current_stats = {
            'agent': 'Initializing...',
            'iteration': 0,
            'urls_processed': 0,
            'urls_to_process': 0,
            'chunks_retrieved': 0,
            'search_queries': 0,
            'retrieve_queries': 0,
            'analysis_results': 0,
            'elapsed_time': 0,
            'last_message': '',
            'current_step': 'Preparing...'
        }
        self.start_time = None
        
    def create_header(self) -> Panel:
        """Creates the CLI header"""
        title = Text("🔬 MetaAnalyst Agent - Interactive CLI", style="bold blue")
        subtitle = Text("Automated Medical Meta-Analysis System", style="italic cyan")
        
        header_content = Align.center(f"{title}\n{subtitle}")
        
        return Panel(
            header_content,
            box=box.DOUBLE,
            border_style="blue",
            padding=(1, 2)
        )
    
    def create_stats_panel(self) -> Panel:
        """Creates the statistics panel"""
        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        table.add_column("Metric", style="bold cyan", min_width=20)
        table.add_column("Value", style="bold white", min_width=15)
        
        # Calculate elapsed time
        elapsed = time.time() - self.start_time if self.start_time else 0
        elapsed_str = f"{elapsed:.1f}s"
        
        table.add_row("🤖 Active Agent:", self.current_stats['agent'])
        table.add_row("🔄 Iteration:", str(self.current_stats['iteration']))
        table.add_row("⏱️  Elapsed Time:", elapsed_str)
        table.add_row("🔍 Search Queries:", str(self.current_stats['search_queries']))
        table.add_row("📥 Retrieve Queries:", str(self.current_stats['retrieve_queries']))
        table.add_row("🌐 URLs Processed:", f"{self.current_stats['urls_processed']}/{self.current_stats['urls_to_process']}")
        table.add_row("📄 Chunks Retrieved:", str(self.current_stats['chunks_retrieved']))
        table.add_row("📊 Analyses Performed:", str(self.current_stats['analysis_results']))
        
        return Panel(
            table,
            title="📈 Real-Time Statistics",
            border_style="green",
            box=box.ROUNDED
        )
    
    def create_progress_panel(self) -> Panel:
        """Creates the progress panel"""
        step_text = Text(f"🔄 {self.current_stats['current_step']}", style="bold yellow")
        
        if self.current_stats['last_message']:
            # Truncate message if too long
            message = self.current_stats['last_message']
            if len(message) > 100:
                message = message[:97] + "..."
            
            last_msg = Text(f"💬 Last action: {message}", style="dim white")
            content = f"{step_text}\n\n{last_msg}"
        else:
            content = step_text
            
        return Panel(
            content,
            title="⚡ Current Progress",
            border_style="yellow",
            box=box.ROUNDED
        )
    
    def create_results_panel(self, insights: List[Dict[str, Any]]) -> Panel:
        """Creates the analysis results panel"""
        if not insights:
            return Panel(
                Align.center("🔍 Waiting for analysis results..."),
                title="📊 Meta-Analysis Results",
                border_style="blue",
                box=box.ROUNDED
            )
        
        content = []
        
        # Show summary
        content.append(f"📈 **Total Insights:** {len(insights)}\n")
        
        # Show the first insights
        for i, insight in enumerate(insights[:3], 1):
            if isinstance(insight, dict) and 'insight' in insight:
                insight_text = insight['insight']
                # Truncate if too long
                if len(insight_text) > 200:
                    insight_text = insight_text[:197] + "..."
                
                content.append(f"**{i}.** {insight_text}\n")
                
                # Show references if available
                if 'references' in insight and insight['references']:
                    refs = insight['references'][:2]  # Show only the first 2
                    ref_text = "; ".join(refs) if isinstance(refs, list) else str(refs)
                    if len(ref_text) > 150:
                        ref_text = ref_text[:147] + "..."
                    content.append(f"   📚 *Sources: {ref_text}*\n")
                
                content.append("")
        
        if len(insights) > 3:
            content.append(f"... and {len(insights) - 3} more insights\n")
        
        markdown_content = "\n".join(content)
        
        return Panel(
            Markdown(markdown_content),
            title="📊 Meta-Analysis Results",
            border_style="magenta",
            box=box.ROUNDED
        )
    
    def update_stats_from_chunk(self, chunk: Dict[str, Any]):
        """Updates statistics based on the received chunk"""
        if isinstance(chunk, dict):
            for agent_name, agent_data in chunk.items():
                if isinstance(agent_data, dict):
                    self.current_stats['agent'] = agent_name.title()
                    
                    # Update specific statistics
                    if 'current_iteration' in agent_data:
                        self.current_stats['iteration'] = agent_data['current_iteration']
                    
                    if 'urls_to_process' in agent_data:
                        self.current_stats['urls_to_process'] = len(agent_data['urls_to_process'])
                    
                    if 'processed_urls' in agent_data:
                        self.current_stats['urls_processed'] = len(agent_data['processed_urls'])
                    
                    if 'retrieved_chunks_count' in agent_data:
                        self.current_stats['chunks_retrieved'] = agent_data['retrieved_chunks_count']
                    
                    if 'previous_search_queries' in agent_data:
                        self.current_stats['search_queries'] = len(agent_data['previous_search_queries'])
                    
                    if 'previous_retrieve_queries' in agent_data:
                        self.current_stats['retrieve_queries'] = len(agent_data['previous_retrieve_queries'])
                    
                    if 'analysis_results' in agent_data:
                        self.current_stats['analysis_results'] = len(agent_data['analysis_results'])
                    
                    # Extract last message
                    if 'messages' in agent_data and agent_data['messages']:
                        last_msg = agent_data['messages'][-1]
                        if hasattr(last_msg, 'content'):
                            content = str(last_msg.content)
                            if isinstance(content, str) and content.strip():
                                self.current_stats['last_message'] = content
                        elif hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                            tool_names = [tc.get('name', 'unknown') for tc in last_msg.tool_calls]
                            self.current_stats['last_message'] = f"Running tools: {', '.join(tool_names)}"
                    
                    # Determine current step
                    self.determine_current_step(agent_name, agent_data)
    
    def determine_current_step(self, agent_name: str, agent_data: Dict[str, Any]):
        """Determines the current step based on the agent and data"""
        if agent_name == "supervisor":
            if not agent_data.get('meta_analysis_pico'):
                self.current_stats['current_step'] = "Defining PICO criteria"
            elif not agent_data.get('urls_to_process') and not agent_data.get('processed_urls'):
                self.current_stats['current_step'] = "Starting bibliographic search"
            elif agent_data.get('urls_to_process'):
                self.current_stats['current_step'] = "Coordinating URL processing"
            elif agent_data.get('retrieved_chunks_count', 0) > 0:
                self.current_stats['current_step'] = "Supervising data analysis"
            else:
                self.current_stats['current_step'] = "Supervising pipeline"
        
        elif agent_name == "researcher":
            self.current_stats['current_step'] = "Researching scientific literature"
        
        elif agent_name == "processor":
            self.current_stats['current_step'] = "Processing and vectorizing content"
        
        elif agent_name == "retriever":
            self.current_stats['current_step'] = "Retrieving relevant chunks"
        
        elif agent_name == "analyzer":
            self.current_stats['current_step'] = "Analyzing data and calculating metrics"
        
        else:
            self.current_stats['current_step'] = f"Running agent {agent_name}"
    
    def run_analysis(self, query: str):
        """Runs the meta-analysis for a query"""
        self.console.clear()
        self.start_time = time.time()
        
        # Show header
        self.console.print(self.create_header())
        
        # Create initial state
        initial_state = MetaAnalysisState(
            user_request=query,
            messages=[{"role": "user", "content": query}],
            urls_to_process=[],
            processed_urls=[],
            current_iteration=1,
            remaining_steps=10,
            meta_analysis_pico={},
            previous_search_queries=[],
            previous_retrieve_queries=[],
            retrieved_chunks_count=0,
            analysis_results=[],
            current_draft="",
            current_draft_iteration=1,
            reviewer_feedbacks=[],
            final_draft=""
        )
        
        # Layout for real-time display
        layout = Layout()
        layout.split_column(
            Layout(name="stats", size=12),
            Layout(name="progress", size=6),
            Layout(name="results", minimum_size=10)
        )
        
        analysis_insights = []
        chunk_count = 0
        
        try:
            with Live(layout, refresh_per_second=2, console=self.console) as live:
                layout["stats"].update(self.create_stats_panel())
                layout["progress"].update(self.create_progress_panel())
                layout["results"].update(self.create_results_panel([]))
                
                for chunk in supervisor_agent.stream(initial_state):
                    chunk_count += 1
                    
                    # Update statistics
                    self.update_stats_from_chunk(chunk)
                    
                    # Extract analysis insights
                    if isinstance(chunk, dict):
                        for agent_name, agent_data in chunk.items():
                            if isinstance(agent_data, dict) and 'analysis_results' in agent_data:
                                analysis_insights = agent_data['analysis_results']
                    
                    # Update panels
                    layout["stats"].update(self.create_stats_panel())
                    layout["progress"].update(self.create_progress_panel())
                    layout["results"].update(self.create_results_panel(analysis_insights))
                    
                    # Safety timeout
                    elapsed = time.time() - self.start_time
                    if elapsed > 60000:  # 10 minutes
                        self.console.print("\n⚠️  [bold red]Timeout reached - Stopping execution[/bold red]")
                        break
                    
                    if chunk_count > 10000:
                        self.console.print("\n⚠️  [bold red]Too many chunks - Possible infinite loop[/bold red]")
                        break
                        
                        # Small pause to avoid overload
                    time.sleep(0.1)
        
        except KeyboardInterrupt:
            self.console.print("\n❌ [bold red]Analysis interrupted by user[/bold red]")
        except Exception as e:
            self.console.print(f"\n❌ [bold red]Error during execution: {str(e)}[/bold red]")
        
        # Show final result
        self.show_final_results(query, analysis_insights, chunk_count)
    
    def show_final_results(self, query: str, insights: List[Dict[str, Any]], chunk_count: int):
        """Shows the final results of the analysis"""
        self.console.print("\n" + "="*80)
        self.console.print(f"🎯 [bold green]ANALYSIS COMPLETED[/bold green] - Total chunks processed: {chunk_count}")
        self.console.print("="*80)
        
        # Execution summary
        summary_table = Table(title="📋 Execution Summary", box=box.ROUNDED)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="white")
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        summary_table.add_row("⏱️ Total Time", f"{elapsed:.2f} seconds")
        summary_table.add_row("🔄 Chunks Processed", str(chunk_count))
        summary_table.add_row("🌐 URLs Processed", str(self.current_stats['urls_processed']))
        summary_table.add_row("📄 Chunks Retrieved", str(self.current_stats['chunks_retrieved']))
        summary_table.add_row("📊 Insights Generated", str(len(insights)))
        
        self.console.print(summary_table)
        
        # Show detailed insights if available
        if insights:
            self.console.print(f"\n📊 [bold blue]META-ANALYSIS RESULTS:[/bold blue]")
            self.console.print(f"Analyzed query: [italic]{query}[/italic]\n")
            
            for i, insight in enumerate(insights, 1):
                if isinstance(insight, dict) and 'insight' in insight:
                    panel_content = []
                    
                    # Main insight
                    insight_text = insight['insight']
                    panel_content.append(insight_text)
                    
                    # References
                    if 'references' in insight and insight['references']:
                        panel_content.append("\n📚 **References:**")
                        refs = insight['references']
                        if isinstance(refs, list):
                            for ref in refs[:3]:  # Show up to 3 references
                                panel_content.append(f"• {ref}")
                        else:
                            panel_content.append(f"• {refs}")
                    
                    # Analysis type
                    if 'analysis_type' in insight:
                        panel_content.append(f"\n🔬 **Type:** {insight['analysis_type']}")
                    
                    content = "\n".join(panel_content)
                    
                    self.console.print(Panel(
                        Markdown(content),
                        title=f"Insight {i}",
                        border_style="blue",
                        box=box.ROUNDED
                    ))
                    self.console.print()
        else:
            self.console.print("⚠️  [yellow]No insight was generated during the analysis[/yellow]")
    
    def interactive_mode(self):
        """Interactive CLI mode"""
        self.console.print(self.create_header())
        
        self.console.print(Panel(
            """
🔍 **How to use:**
• Type your medical meta-analysis query
• Use 'quit' or 'exit' to leave
• Use 'help' to see example queries
• Use Ctrl+C to interrupt an ongoing analysis

💡 **Example queries:**
• "Amiodarone vs beta-blockers for atrial fibrillation"
• "Statins vs placebo for cardiovascular prevention"
• "ACE inhibitors vs ARBs for hypertension"
            """,
            title="🚀 Interactive Mode",
            border_style="cyan"
        ))
        
        while True:
            try:
                query = self.console.input("\n🔬 [bold cyan]Enter your meta-analysis query:[/bold cyan] ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    self.console.print("👋 [bold green]Thank you for using MetaAnalyst CLI![/bold green]")
                    break
                
                if query.lower() == 'help':
                    self.show_help()
                    continue
                
                # Run analysis
                self.console.print(f"\n🚀 [bold green]Starting meta-analysis for:[/bold green] {query}")
                self.run_analysis(query)
                
                # Ask if want to continue
                continue_choice = self.console.input("\n🤔 [bold yellow]Do you want to run another analysis? (y/n):[/bold yellow] ").strip().lower()
                if continue_choice in ['n', 'no']:
                    self.console.print("👋 [bold green]Thank you for using MetaAnalyst CLI![/bold green]")
                    break
                
            except KeyboardInterrupt:
                self.console.print("\n\n👋 [bold green]Thank you for using MetaAnalyst CLI![/bold green]")
                break
            except EOFError:
                break
    
    def show_help(self):
        """Shows help with example queries"""
        help_content = """
📖 **MetaAnalyst CLI Usage Guide**

🎯 **Supported Query Types:**

**1. Drug Comparisons:**
• "Aspirin vs clopidogrel for stroke prevention"
• "Metformin vs insulin for diabetes type 2"
• "Warfarin vs DOACs for atrial fibrillation"

**2. Interventions vs Control:**
• "Exercise therapy vs placebo for depression"
• "Mediterranean diet vs low-fat diet for cardiovascular health"
• "Cognitive behavioral therapy vs pharmacotherapy for anxiety"

**3. Medical Procedures:**
• "Laparoscopic vs open surgery for gallbladder removal"
• "Stents vs medical therapy for coronary artery disease"
• "Robotic vs conventional surgery for prostate cancer"

**4. Treatment Strategies:**
• "Early vs delayed antiretroviral therapy for HIV"
• "Intensive vs standard blood pressure control"
• "Conservative vs surgical management for knee osteoarthritis"

💡 **Tips for Better Results:**
• Use specific medical terms in English
• Include the condition/disease at the end of the query
• Be specific about the population (e.g., "elderly patients", "pediatric")
• Mention important outcomes (e.g., "mortality", "quality of life")

⏱️  **Average Execution Time:** 5-10 minutes depending on complexity
📊 **Results:** Insights based on current scientific literature
        """
        
        self.console.print(Panel(
            Markdown(help_content),
            title="📚 Help - MetaAnalyst CLI",
            border_style="blue",
            box=box.ROUNDED
        ))

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Interactive CLI for Medical Meta-Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python cli.py                                    # Interactive mode
  python cli.py -q "Aspirin vs placebo for MI"     # Single query
  python cli.py --help                             # Show help
        """
    )
    
    parser.add_argument(
        "-q", "--query",
        type=str,
        help="Specific query for analysis (non-interactive mode)"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="MetaAnalyst CLI v1.0.0"
    )
    
    args = parser.parse_args()
    
    # Check if dependencies are installed
    try:
        from rich.console import Console
    except ImportError:
        print("❌ Error: Dependency 'rich' not found.")
        print("📦 Install with: pip install rich")
        sys.exit(1)
    
    # Check environment variables
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY not found.")
        print("🔑 Set your API key in the .env file")
        sys.exit(1)
    
    cli = MetaAnalysisCLI()
    
    if args.query:
        # Single query mode
        cli.console.print(f"🚀 [bold green]Running analysis for:[/bold green] {args.query}")
        cli.run_analysis(args.query)
    else:
        # Interactive mode
        cli.interactive_mode()

if __name__ == "__main__":
    main()
