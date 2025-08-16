from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.columns import Columns
from typing import List, Dict, Any
import textwrap

class ConversationDisplay:
    """Display roundtable conversations in a formatted way."""
    
    def __init__(self):
        self.console = Console()
    
    def display_roundtable_conversation(self, messages: List[Any]):
        """Display the roundtable conversation in a formatted table."""
        if not messages:
            self.console.print("[yellow]No conversation messages to display[/yellow]")
            return
        
        table = Table(
            title="🏛️ Roundtable Discussion",
            show_header=True,
            header_style="bold magenta",
            show_lines=True,
            expand=True
        )
        
        table.add_column("Round", style="cyan", width=8, justify="center")
        table.add_column("Role", style="green", width=12)
        table.add_column("Statement", style="white", width=100)
        
        role_styles = {
            'moderator': 'blue',
            'prosecutor': 'red',
            'defender': 'green',
            'judge': 'yellow',
            'jury': 'magenta'
        }
        
        for msg in messages:
            if hasattr(msg, 'additional_kwargs'):
                kwargs = msg.additional_kwargs
                round_num = str(kwargs.get('round', 0))
                role = kwargs.get('role', 'unknown')
                content = msg.content
                
                # Truncate long content but keep it readable
                if len(content) > 300:
                    content = textwrap.fill(content[:297] + "...", width=95)
                else:
                    content = textwrap.fill(content, width=95)
                
                # Apply role-specific styling
                role_display = Text(role.title(), style=role_styles.get(role, 'white'))
                
                table.add_row(round_num, role_display, content)
        
        self.console.print("\n")
        self.console.print(table)
        self.console.print("\n")
    
    def display_errors_summary(self, errors: List[Dict[str, Any]]):
        """Display identified errors in a structured panel."""
        if not errors:
            self.console.print(Panel(
                "✅ [green]No logical errors identified[/green]",
                title="Analysis Results",
                style="green"
            ))
            return
        
        # Group errors by severity
        high_errors = [e for e in errors if e.get('severity') == 'high']
        medium_errors = [e for e in errors if e.get('severity') == 'medium']
        low_errors = [e for e in errors if e.get('severity') == 'low']
        
        error_text = ""
        
        if high_errors:
            error_text += "[bold red]High Severity Issues:[/bold red]\n"
            for i, error in enumerate(high_errors, 1):
                desc = textwrap.fill(error.get('description', ''), width=70, subsequent_indent='     ')
                error_text += f"  {i}. {desc}\n"
            error_text += "\n"
        
        if medium_errors:
            error_text += "[bold yellow]Medium Severity Issues:[/bold yellow]\n"
            for i, error in enumerate(medium_errors, 1):
                desc = textwrap.fill(error.get('description', ''), width=70, subsequent_indent='     ')
                error_text += f"  {i}. {desc}\n"
            error_text += "\n"
        
        if low_errors:
            error_text += "[bold cyan]Low Severity Issues:[/bold cyan]\n"
            for i, error in enumerate(low_errors, 1):
                desc = textwrap.fill(error.get('description', ''), width=70, subsequent_indent='     ')
                error_text += f"  {i}. {desc}\n"
        
        self.console.print(Panel(
            error_text.strip(),
            title=f"🔍 Identified Errors ({len(errors)} total)",
            border_style="red"
        ))
    
    def display_verdict(self, verdict: Dict[str, Any]):
        """Display the final verdict."""
        if not verdict:
            return
        
        score = verdict.get('score', 'N/A')
        strengths = verdict.get('strengths', [])
        weaknesses = verdict.get('weaknesses', [])
        recommendations = verdict.get('recommendations', [])
        
        verdict_text = f"[bold]Overall Score:[/bold] {score}/10\n\n"
        
        if strengths:
            verdict_text += "[green]Strengths:[/green]\n"
            for s in strengths:
                verdict_text += f"  • {s}\n"
            verdict_text += "\n"
        
        if weaknesses:
            verdict_text += "[red]Weaknesses:[/red]\n"
            for w in weaknesses:
                verdict_text += f"  • {w}\n"
            verdict_text += "\n"
        
        if recommendations:
            verdict_text += "[yellow]Recommendations:[/yellow]\n"
            for r in recommendations:
                verdict_text += f"  • {r}\n"
        
        self.console.print(Panel(
            verdict_text.strip(),
            title="⚖️ Final Verdict",
            border_style="magenta"
        ))
    
    def display_complete_analysis(self, result: Dict[str, Any]):
        """Display complete analysis including conversation, errors, and verdict."""
        self.console.print("\n" + "="*100 + "\n", style="bold blue")
        self.console.print("[bold blue]📊 COMPLETE ANALYSIS REPORT[/bold blue]", justify="center")
        self.console.print("="*100 + "\n", style="bold blue")
        
        # Display conversation
        if 'messages' in result:
            self.display_roundtable_conversation(result['messages'])
        
        # Display errors
        if 'identified_errors' in result:
            self.display_errors_summary(result['identified_errors'])
        
        # Display verdict
        if 'final_verdict' in result:
            self.display_verdict(result['final_verdict'])
        
        # Display timing information if available
        if 'timings' in result:
            self.display_timings(result['timings'])
    
    def display_timings(self, timings: Dict[str, float]):
        """Display timing information."""
        timing_text = ""
        for key, value in timings.items():
            timing_text += f"{key.replace('_', ' ').title()}: {value:.2f}ms\n"
        
        self.console.print(Panel(
            timing_text.strip(),
            title="⏱️ Performance Metrics",
            style="cyan"
        ))