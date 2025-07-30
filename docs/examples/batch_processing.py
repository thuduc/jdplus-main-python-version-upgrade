"""Batch processing example for multiple time series."""

import numpy as np
import pandas as pd
from typing import Dict, List
import time
from jdemetra_py.toolkit.timeseries import TsData, TsPeriod, TsFrequency
from jdemetra_py.sa.tramoseats import TramoSeatsSpecification, TramoSeatsProcessor
from jdemetra_py.workspace import Workspace, SAItem
from jdemetra_py.io import TsCollection, ExcelDataProvider
from jdemetra_py.optimization import parallel_map, enable_numba
from jdemetra_py.sa.diagnostics import QualityMeasures

def generate_sample_series(n_series: int = 10, n_years: int = 5) -> Dict[str, TsData]:
    """Generate sample time series for testing."""
    np.random.seed(42)
    series_dict = {}
    
    n_obs = n_years * 12
    start = TsPeriod.of(TsFrequency.MONTHLY, 2019, 0)
    
    for i in range(n_series):
        # Generate with different characteristics
        t = np.arange(n_obs)
        
        # Varying trend
        trend_slope = 0.2 + 0.3 * i / n_series
        trend = 100 + trend_slope * t
        
        # Varying seasonal amplitude
        seasonal_amp = 5 + 10 * i / n_series
        seasonal = seasonal_amp * np.sin(2 * np.pi * t / 12 + i)
        
        # Varying noise
        noise_level = 1 + 2 * i / n_series
        irregular = np.random.randn(n_obs) * noise_level
        
        # Combine
        values = trend + seasonal + irregular
        
        # Add some outliers
        if i % 3 == 0:
            # Add level shift
            values[30:] += 10
        elif i % 3 == 1:
            # Add additive outlier
            values[25] += 20
        
        series_name = f"Series_{i+1:02d}"
        series_dict[series_name] = TsData.of(start, values)
    
    return series_dict

def process_with_workspace(series_dict: Dict[str, TsData]) -> Workspace:
    """Process series using workspace approach."""
    print("Processing with Workspace approach...")
    
    # Create workspace
    workspace = Workspace("Batch Processing Example")
    
    # Add all series with same specification
    spec = TramoSeatsSpecification.rsa5()
    
    for name, series in series_dict.items():
        item = SAItem(name, series, spec)
        item.metadata = {
            'category': 'Economic' if 'Series_0' in name else 'Financial',
            'importance': 'High' if int(name.split('_')[1]) <= 5 else 'Medium'
        }
        workspace.add_sa_item(item)
    
    # Process all items
    start_time = time.time()
    processor = TramoSeatsProcessor()
    results = workspace.process_all(processor)
    processing_time = time.time() - start_time
    
    # Summary
    successful = sum(1 for success in results.values() if success)
    print(f"  Processed {successful}/{len(results)} series successfully")
    print(f"  Total time: {processing_time:.2f} seconds")
    print(f"  Average time per series: {processing_time/len(results):.2f} seconds")
    
    return workspace

def process_with_parallel(series_dict: Dict[str, TsData]) -> List:
    """Process series using parallel processing."""
    print("\nProcessing with parallel approach...")
    
    # Enable Numba for faster processing
    enable_numba()
    
    # Define processing function
    def process_single(name_series_tuple):
        name, series = name_series_tuple
        spec = TramoSeatsSpecification.rsa5()
        processor = TramoSeatsProcessor()
        
        try:
            results = processor.process(series)
            quality = QualityMeasures(results.decomposition)
            return {
                'name': name,
                'success': True,
                'results': results,
                'q_stat': quality.q()
            }
        except Exception as e:
            return {
                'name': name,
                'success': False,
                'error': str(e)
            }
    
    # Process in parallel
    start_time = time.time()
    results = parallel_map(
        process_single, 
        list(series_dict.items()),
        n_workers=4
    )
    processing_time = time.time() - start_time
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    print(f"  Processed {successful}/{len(results)} series successfully")
    print(f"  Total time: {processing_time:.2f} seconds")
    print(f"  Average time per series: {processing_time/len(results):.2f} seconds")
    
    return results

def generate_quality_report(workspace: Workspace) -> pd.DataFrame:
    """Generate quality report for all series."""
    report_data = []
    
    for item_id in workspace.list_sa_items():
        item = workspace.get_sa_item(item_id)
        
        if item.results is not None:
            decomp = item.results.decomposition
            quality = QualityMeasures(decomp)
            
            # Calculate additional metrics
            sa = decomp.seasonally_adjusted
            seasonal = decomp.seasonal
            
            report_data.append({
                'Series': item.name,
                'Category': item.metadata.get('category', 'Unknown'),
                'Importance': item.metadata.get('importance', 'Unknown'),
                'Observations': sa.length(),
                'Q-Stat': round(quality.q(), 3),
                'M1': round(quality.m1(), 3),
                'M7': round(quality.m7(), 3),
                'SA Mean': round(sa.average(), 2),
                'Seasonal Range': round(np.ptp(seasonal.values), 2),
                'Status': 'Good' if quality.q() < 1.0 else 'Review needed'
            })
    
    return pd.DataFrame(report_data)

def save_results(workspace: Workspace, output_dir: str = '.'):
    """Save all results to files."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save workspace
    workspace.save(f"{output_dir}/batch_workspace.json")
    print(f"\nSaved workspace to {output_dir}/batch_workspace.json")
    
    # Create collection of SA series
    collection = TsCollection("Seasonally Adjusted Series")
    
    for item_id in workspace.list_sa_items():
        item = workspace.get_sa_item(item_id)
        if item.results is not None:
            sa = item.results.decomposition.seasonally_adjusted
            collection.add(f"{item.name}_SA", sa)
    
    # Save to Excel
    excel = ExcelDataProvider()
    excel.write(collection, f"{output_dir}/batch_sa_results.xlsx")
    print(f"Saved SA series to {output_dir}/batch_sa_results.xlsx")
    
    # Save quality report
    report_df = generate_quality_report(workspace)
    report_df.to_csv(f"{output_dir}/batch_quality_report.csv", index=False)
    print(f"Saved quality report to {output_dir}/batch_quality_report.csv")
    
    # Generate summary plots
    from jdemetra_py.visualization import plot_series
    import matplotlib.pyplot as plt
    
    # Plot first 4 series
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, item_id in enumerate(workspace.list_sa_items()[:4]):
        item = workspace.get_sa_item(item_id)
        if item.results is not None:
            ax = axes[i]
            
            orig = item.series
            sa = item.results.decomposition.seasonally_adjusted
            
            dates = pd.date_range('2019-01', periods=orig.length(), freq='M')
            ax.plot(dates, orig.values, label='Original', alpha=0.7)
            ax.plot(dates, sa.values, label='SA', linewidth=2)
            ax.set_title(item.name)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/batch_sample_plots.png", dpi=300)
    print(f"Saved sample plots to {output_dir}/batch_sample_plots.png")

def main():
    """Run batch processing example."""
    print("Batch Processing Example")
    print("=" * 50)
    
    # Generate sample data
    print("\nGenerating sample time series...")
    series_dict = generate_sample_series(n_series=20, n_years=5)
    print(f"Generated {len(series_dict)} time series")
    
    # Method 1: Workspace approach
    workspace = process_with_workspace(series_dict)
    
    # Method 2: Parallel processing
    parallel_results = process_with_parallel(series_dict)
    
    # Compare methods
    print("\nComparison:")
    print("  Workspace approach is good for:")
    print("    - Managing related series together")
    print("    - Saving/loading entire projects")
    print("    - Tracking metadata and results")
    print("  Parallel approach is good for:")
    print("    - Maximum processing speed")
    print("    - Simple batch operations")
    print("    - Custom processing logic")
    
    # Generate quality report
    print("\nQuality Report:")
    report_df = generate_quality_report(workspace)
    print(report_df.head(10))
    
    # Quality summary
    print("\nQuality Summary:")
    print(f"  Excellent (Q < 1.0): {len(report_df[report_df['Q-Stat'] < 1.0])}")
    print(f"  Good (Q < 1.5): {len(report_df[(report_df['Q-Stat'] >= 1.0) & (report_df['Q-Stat'] < 1.5)])}")
    print(f"  Review needed (Q >= 1.5): {len(report_df[report_df['Q-Stat'] >= 1.5])}")
    
    # Save all results
    save_results(workspace, output_dir='batch_results')
    
    print("\nBatch processing completed!")

if __name__ == '__main__':
    main()