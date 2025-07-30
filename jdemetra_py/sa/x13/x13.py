"""X-13ARIMA-SEATS wrapper implementation."""

import os
import tempfile
import subprocess
import json
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
import logging

from ...toolkit.timeseries import TsData, TsPeriod, TsDomain
from ..base import SaProcessor, SaResults
from ..base.results import SeriesDecomposition, DecompositionMode
from .specification import X13Specification


class X13ArimaSeatsProcessor(SaProcessor):
    """X-13ARIMA-SEATS processor wrapper."""
    
    def __init__(self, specification: Optional[X13Specification] = None,
                 x13_path: Optional[str] = None):
        """Initialize X-13ARIMA-SEATS processor.
        
        Args:
            specification: X-13 specification (uses default if None)
            x13_path: Path to X-13ARIMA-SEATS executable
        """
        if specification is None:
            specification = X13Specification.rsa1()
        
        self.spec = specification
        self.x13_path = x13_path or self._find_x13_executable()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if not self.x13_path or not os.path.exists(self.x13_path):
            raise RuntimeError("X-13ARIMA-SEATS executable not found")
    
    def process(self, series: TsData) -> 'X13Results':
        """Process series with X-13ARIMA-SEATS.
        
        Args:
            series: Input time series
            
        Returns:
            X-13 results
        """
        self.logger.info(f"Processing series with X-13ARIMA-SEATS (length={series.length})")
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write input files
            spec_file = os.path.join(temp_dir, "spec.spc")
            data_file = os.path.join(temp_dir, "data.dat")
            
            self._write_spec_file(spec_file, series, data_file)
            self._write_data_file(data_file, series)
            
            # Run X-13
            self.logger.info("Running X-13ARIMA-SEATS...")
            self._run_x13(spec_file, temp_dir)
            
            # Read results
            results = self._read_results(temp_dir, series)
            
            # Create X13Results
            x13_results = X13Results(
                specification=self.spec,
                decomposition=results['decomposition'],
                output_files=results.get('output_files', {}),
                diagnostics_output=results.get('diagnostics', {})
            )
            
            # Compute diagnostics
            x13_results.compute_diagnostics()
            
            self.logger.info("X-13ARIMA-SEATS processing complete")
            
            return x13_results
    
    def _find_x13_executable(self) -> Optional[str]:
        """Find X-13ARIMA-SEATS executable."""
        # Common locations
        possible_paths = [
            "x13as",  # In PATH
            "/usr/local/bin/x13as",
            "/usr/bin/x13as",
            "C:\\Program Files\\x13as\\x13as.exe",
            "C:\\x13as\\x13as.exe"
        ]
        
        # Check environment variable
        env_path = os.environ.get("X13PATH")
        if env_path:
            possible_paths.insert(0, env_path)
        
        # Find first existing
        for path in possible_paths:
            if os.path.exists(path):
                return path
            
            # Try with shutil.which
            import shutil
            found = shutil.which(path)
            if found:
                return found
        
        return None
    
    def _write_spec_file(self, spec_file: str, series: TsData, data_file: str):
        """Write X-13 specification file."""
        lines = []
        
        # Series section
        lines.append("series{")
        lines.append(f"  file = '{os.path.basename(data_file)}'")
        lines.append(f"  format = '1f12.2'")
        lines.append(f"  period = {series.domain.frequency.periods_per_year}")
        
        # Start date
        start_year = series.start.year
        start_period = series.start.position + 1
        lines.append(f"  start = {start_year}.{start_period}")
        lines.append("}")
        
        # Transform section
        if self.spec.regression and self.spec.regression.transform:
            transform = self.spec.regression.transform
            if transform.function != "none":
                lines.append("transform{")
                if transform.function == "auto":
                    lines.append("  function = auto")
                elif transform.function == "log":
                    lines.append("  function = log")
                lines.append("}")
        
        # ARIMA section
        if self.spec.regression and self.spec.regression.arima:
            arima = self.spec.regression.arima
            lines.append("arima{")
            if arima.model:
                lines.append(f"  model = {arima.model}")
            lines.append("}")
        
        # Regression section
        if self.spec.regression and self.spec.regression.variables:
            vars = self.spec.regression.variables
            lines.append("regression{")
            
            variables = []
            if vars.td:
                variables.append(vars.td)
            if vars.easter:
                variables.append(f"easter[{vars.easter}]")
            if vars.outlier:
                variables.extend(vars.outlier)
            
            if variables:
                lines.append(f"  variables = ({' '.join(variables)})")
            
            lines.append("}")
        
        # Outlier section
        if self.spec.regression and self.spec.regression.outliers and \
           self.spec.regression.outliers.enabled:
            lines.append("outlier{")
            if self.spec.regression.outliers.types:
                types = " ".join(self.spec.regression.outliers.types).lower()
                lines.append(f"  types = ({types})")
            lines.append("}")
        
        # X11 or SEATS section
        if self.spec.x11:
            lines.append("x11{")
            if self.spec.x11.mode:
                lines.append(f"  mode = {self.spec.x11.mode}")
            lines.append("  save = (d10 d11 d12 d13)")  # Components
            lines.append("}")
        elif self.spec.seats:
            lines.append("seats{")
            lines.append("  save = (s10 s11 s12 s13)")  # Components
            lines.append("}")
        
        # Forecast section
        lines.append("forecast{")
        lines.append("  maxlead = 12")
        lines.append("}")
        
        # Write file
        with open(spec_file, 'w') as f:
            f.write('\n'.join(lines))
    
    def _write_data_file(self, data_file: str, series: TsData):
        """Write data file for X-13."""
        with open(data_file, 'w') as f:
            for value in series.values:
                if np.isnan(value):
                    f.write(" -99999.00\n")
                else:
                    f.write(f" {value:10.2f}\n")
    
    def _run_x13(self, spec_file: str, work_dir: str):
        """Run X-13ARIMA-SEATS executable."""
        # Change to working directory
        original_dir = os.getcwd()
        os.chdir(work_dir)
        
        try:
            # Run X-13
            cmd = [self.x13_path, os.path.basename(spec_file)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"X-13 error: {result.stderr}")
                raise RuntimeError(f"X-13 execution failed: {result.stderr}")
            
            # Log output
            if result.stdout:
                self.logger.debug(f"X-13 output: {result.stdout}")
            
        finally:
            os.chdir(original_dir)
    
    def _read_results(self, work_dir: str, original_series: TsData) -> Dict[str, Any]:
        """Read X-13 output files."""
        results = {
            'output_files': {},
            'diagnostics': {}
        }
        
        # Create decomposition
        decomp = SeriesDecomposition(mode=DecompositionMode.ADDITIVE)
        decomp.series = original_series
        
        # Read components based on method
        if self.spec.x11:
            # X-11 components
            components = {
                'd11': 'seasonally_adjusted',
                'd12': 'trend',
                'd10': 'seasonal',
                'd13': 'irregular'
            }
        else:
            # SEATS components
            components = {
                's11': 'seasonally_adjusted',
                's12': 'trend',
                's10': 'seasonal',
                's13': 'irregular'
            }
        
        # Read each component
        for file_suffix, component_name in components.items():
            file_path = os.path.join(work_dir, f"spec.{file_suffix}")
            if os.path.exists(file_path):
                data = self._read_x13_output_file(file_path)
                if data is not None:
                    # Create TsData
                    ts = TsData.of(original_series.start, data)
                    setattr(decomp, component_name, ts)
        
        # Read diagnostics files
        diag_files = ['out', 'err', 'log']
        for suffix in diag_files:
            file_path = os.path.join(work_dir, f"spec.{suffix}")
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    results['output_files'][suffix] = f.read()
        
        # Parse diagnostics from output
        if 'out' in results['output_files']:
            results['diagnostics'] = self._parse_x13_diagnostics(
                results['output_files']['out']
            )
        
        results['decomposition'] = decomp
        
        return results
    
    def _read_x13_output_file(self, file_path: str) -> Optional[np.ndarray]:
        """Read X-13 output data file."""
        try:
            # X-13 output files have specific format
            data = []
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('-'):
                        try:
                            # Parse values (may have multiple per line)
                            values = line.split()
                            for val in values:
                                if val != '-99999.00':
                                    data.append(float(val))
                                else:
                                    data.append(np.nan)
                        except ValueError:
                            continue
            
            return np.array(data) if data else None
            
        except Exception as e:
            self.logger.error(f"Error reading {file_path}: {e}")
            return None
    
    def _parse_x13_diagnostics(self, output: str) -> Dict[str, Any]:
        """Parse diagnostics from X-13 output."""
        diagnostics = {}
        
        # Parse key statistics (simplified)
        lines = output.split('\n')
        
        for i, line in enumerate(lines):
            # Look for specific diagnostic sections
            if "ARIMA Model" in line:
                diagnostics['model'] = line.strip()
            
            elif "Ljung-Box" in line:
                # Try to extract p-value
                for j in range(i, min(i+5, len(lines))):
                    if "Prob" in lines[j] or "p-value" in lines[j]:
                        parts = lines[j].split()
                        for part in parts:
                            try:
                                pval = float(part)
                                if 0 <= pval <= 1:
                                    diagnostics['ljung_box_pvalue'] = pval
                                    break
                            except:
                                continue
            
            elif "outliers were identified" in line:
                # Extract outlier count
                parts = line.split()
                for j, part in enumerate(parts):
                    if part.isdigit():
                        diagnostics['n_outliers'] = int(part)
                        break
        
        return diagnostics


class X13Results(SaResults):
    """Results from X-13ARIMA-SEATS."""
    
    def __init__(self, specification: X13Specification,
                 decomposition, output_files: Dict[str, str],
                 diagnostics_output: Dict[str, Any]):
        """Initialize X-13 results.
        
        Args:
            specification: Specification used
            decomposition: Series decomposition
            output_files: Raw output files from X-13
            diagnostics_output: Parsed diagnostics
        """
        super().__init__(specification, decomposition)
        self.output_files = output_files
        self.diagnostics_output = diagnostics_output
    
    def compute_diagnostics(self):
        """Compute diagnostics for X-13 results."""
        # Basic decomposition diagnostics
        super().compute_diagnostics()
        
        # Add X-13 specific diagnostics
        self.diagnostics['x13'] = self.diagnostics_output
        
        # Model info
        if 'model' in self.diagnostics_output:
            self.diagnostics['model'] = self.diagnostics_output['model']
        
        # Outliers
        if 'n_outliers' in self.diagnostics_output:
            self.diagnostics['outliers'] = {
                'count': self.diagnostics_output['n_outliers']
            }
        
        # Test statistics
        if 'ljung_box_pvalue' in self.diagnostics_output:
            self.diagnostics['residual_tests'] = {
                'ljung_box': {
                    'pvalue': self.diagnostics_output['ljung_box_pvalue']
                }
            }
    
    def summary(self) -> str:
        """Get summary of results."""
        lines = ["X-13ARIMA-SEATS Results Summary"]
        lines.append("=" * 40)
        
        # Series info
        lines.append(f"\nSeries length: {self.decomposition.series.length}")
        lines.append(f"Period: {self.decomposition.series.domain.frequency.periods_per_year}")
        
        # Method
        if self.specification.x11:
            lines.append("\nMethod: X-11")
            lines.append(f"  Mode: {self.specification.x11.mode}")
        elif self.specification.seats:
            lines.append("\nMethod: SEATS")
        
        # Model
        if 'model' in self.diagnostics:
            lines.append(f"\nModel: {self.diagnostics['model']}")
        
        # Outliers
        if 'outliers' in self.diagnostics:
            lines.append(f"\nOutliers detected: {self.diagnostics['outliers']['count']}")
        
        # Decomposition
        lines.append("\nDecomposition:")
        lines.append(f"  Mode: {self.decomposition.mode.name}")
        
        # Components available
        components = []
        if self.decomposition.trend is not None:
            components.append("Trend")
        if self.decomposition.seasonal is not None:
            components.append("Seasonal")
        if self.decomposition.irregular is not None:
            components.append("Irregular")
        if self.decomposition.seasonally_adjusted is not None:
            components.append("Seasonally Adjusted")
        
        lines.append(f"  Components: {', '.join(components)}")
        
        # Key diagnostics
        lines.append("\nKey Diagnostics:")
        if 'residual_tests' in self.diagnostics:
            tests = self.diagnostics['residual_tests']
            if 'ljung_box' in tests:
                lines.append(f"  Ljung-Box test p-value: {tests['ljung_box']['pvalue']:.4f}")
        
        return "\n".join(lines)
    
    def get_output(self, file_type: str) -> Optional[str]:
        """Get raw X-13 output file content.
        
        Args:
            file_type: Type of output ('out', 'err', 'log')
            
        Returns:
            File content or None
        """
        return self.output_files.get(file_type)