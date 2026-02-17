"""
Report Export Module

Handles the serialization and saving of system reports to disk.
Supports JSON and CSV formats. Optional Matplotlib plotting.
"""

import json
import csv
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

class ReportExporter:
    def __init__(self, output_dir: str = "report_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def export_json(self, data: Dict[str, Any], filename: str = None) -> str:
        """Save dictionary as JSON."""
        if not filename:
            filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, default=str)
            
        print(f"📄 JSON Report saved to: {filepath}")
        return str(filepath)

    def export_csv(self, flat_data: List[Dict[str, Any]], filename: str = None) -> str:
        """Save list of flat dictionaries as CSV."""
        if not flat_data:
            print("⚠️ No data to export to CSV.")
            return ""
            
        if not filename:
             filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
             
        filepath = self.output_dir / filename
        
        keys = flat_data[0].keys()
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(flat_data)
            
        print(f"📄 CSV Report saved to: {filepath}")
        return str(filepath)
    
    def try_generate_chart(self, data: Dict[str, Any]):
        """
        Attempt to generate a chart if matplotlib is installed.
        Does NOT crash if matplotlib is missing.
        """
        try:
            import matplotlib.pyplot as plt
            
            # Example: Bar chart of accuracy distribution
            if "accuracy_distribution" in data:
                acc = data["accuracy_distribution"]
                # Extract simple counts
                # Data structure usually: {'0_matches': {'count': 10, ...}, ...}
                
                labels = []
                values = []
                
                for k in range(7):
                    key = f"{k}_matches"
                    if key in acc:
                        labels.append(str(k))
                        values.append(acc[key].get('pct', 0)) # Percentage
                
                plt.figure(figsize=(10, 6))
                plt.bar(labels, values, color='#444444')
                plt.title('Match Distribution vs Random (Engine Audit)')
                plt.xlabel('Matches')
                plt.ylabel('Percentage (%)')
                plt.grid(axis='y', alpha=0.3)
                
                filename = f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                filepath = self.output_dir / filename
                plt.savefig(filepath)
                plt.close()
                print(f"📊 Chart saved to: {filepath}")
                
        except ImportError:
            print("ℹ️ Matplotlib not found. Skipping chart generation.")
        except Exception as e:
            print(f"⚠️ Failed to generate chart: {e}")

if __name__ == "__main__":
    # Test
    exporter = ReportExporter()
    test_data = {"test": "data", "accuracy_distribution": {"3_matches": {"pct": 3.0}}}
    exporter.export_json(test_data, "test_report.json")
    exporter.try_generate_chart(test_data)
