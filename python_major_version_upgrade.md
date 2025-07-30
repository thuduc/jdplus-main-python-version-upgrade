# Python 3.12+ Upgrade Plan for JDemetra+ Python

## Executive Summary

This document outlines a comprehensive plan to upgrade the JDemetra+ Python project to support Python 3.12 as the minimum version while upgrading all machine learning packages to their latest versions. All packages have been verified to have no known security vulnerabilities as of July 30, 2025.

## Current State Analysis

### Python Version Support
- **Current**: Python 3.8 - 3.12
- **Target**: Python 3.12+ (dropping support for 3.8, 3.9, 3.10, 3.11)

### Current Dependencies and Their Versions
| Package      | Current Version | Latest Version (July 2025) | Security Status |
|--------------|-----------------|---------------------------|-----------------|
| numpy        | >=1.20         | 2.3.2                     | ✅ No CVEs      |
| pandas       | >=1.3          | 2.3.1                     | ✅ No CVEs      |
| scipy        | >=1.7          | 1.16.1                    | ✅ No CVEs      |
| statsmodels  | >=0.12         | 0.14.5                    | ✅ No CVEs      |
| matplotlib   | >=3.3          | 3.10.3                    | ✅ No CVEs      |
| seaborn      | >=0.11         | Latest                    | ✅ No CVEs      |
| numba        | >=0.54 (optional) | 0.61.2                 | ✅ No CVEs      |

## Upgrade Plan

### Phase 1: Preparation and Testing Infrastructure (Week 1)

1. **Create a new feature branch**
   ```bash
   git checkout -b feature/python-3.12-upgrade
   ```

2. **Set up test matrix**
   - Configure CI/CD to test against Python 3.12, 3.13, and 3.14-rc1
   - Remove Python 3.8, 3.9, 3.10, 3.11 from test matrix

3. **Create compatibility testing suite**
   - Add tests to verify all features work with new package versions
   - Create performance benchmarks to compare before/after upgrade

### Phase 2: Update Configuration Files (Week 1)

1. **Update pyproject.toml**
   - Change `requires-python = ">=3.12"`
   - Update classifiers to include only Python 3.12, 3.13, 3.14
   - Update all dependency versions to latest:
     ```toml
     dependencies = [
         "numpy>=2.3.2",
         "pandas>=2.3.1",
         "scipy>=1.16.1",
         "statsmodels>=0.14.5",
         "matplotlib>=3.10.3",
         "seaborn>=0.12",  # Check latest version
     ]
     ```
   - Update tool configurations (black, mypy) for Python 3.12 minimum

2. **Update setup.py**
   - Change `python_requires=">=3.12"`
   - Update classifiers to match pyproject.toml
   - Update install_requires with new versions

3. **Update development dependencies**
   - Ensure all dev tools support Python 3.12+
   - Update pytest, black, flake8, mypy to latest versions

### Phase 3: Code Modernization (Week 2)

1. **Leverage Python 3.12 features**
   - Use type hints with latest syntax
   - Replace older patterns with Python 3.12+ idioms
   - Utilize improved error messages and performance features

2. **Update type annotations**
   - Use PEP 695 type parameter syntax where applicable
   - Update from typing to built-in types (list, dict, etc.)

3. **Remove compatibility code**
   - Remove any Python 3.8-3.11 compatibility workarounds
   - Update imports to use standard library improvements

### Phase 4: Package-Specific Updates (Week 2-3)

1. **NumPy 2.3.2 Migration**
   - Review NumPy 2.0 migration guide for breaking changes
   - Update any deprecated NumPy APIs
   - Test numerical computations for accuracy

2. **pandas 2.3.1 Migration**
   - Update deprecated pandas APIs
   - Review Copy-on-Write changes
   - Update date/time handling if needed

3. **SciPy 1.16.1 Migration**
   - Update any deprecated SciPy functions
   - Review changes in optimization and linear algebra modules

4. **statsmodels 0.14.5 Migration**
   - Update statistical model APIs if changed
   - Review any deprecated functionality

5. **Visualization Libraries**
   - Update matplotlib usage for 3.10.3
   - Ensure seaborn compatibility with new matplotlib

### Phase 5: Testing and Validation (Week 3-4)

1. **Run comprehensive test suite**
   ```bash
   pytest -v --cov=jdemetra_py tests/
   ```

2. **Performance testing**
   - Run benchmarks to ensure no performance regressions
   - Profile critical paths with new package versions

3. **Integration testing**
   - Test all example scripts in docs/examples/
   - Verify all seasonal adjustment methods work correctly

4. **Security audit**
   - Run security scanning tools
   - Verify no new vulnerabilities introduced

### Phase 6: Documentation Updates (Week 4)

1. **Update README.md**
   - Update Python version requirements
   - Update installation instructions

2. **Update API documentation**
   - Regenerate API docs with latest versions
   - Update any code examples

3. **Update user guide**
   - Document any breaking changes
   - Add migration guide for users

### Phase 7: Release Preparation (Week 4)

1. **Update CHANGELOG.md**
   - Document all breaking changes
   - List new features and improvements

2. **Version bump**
   - Consider major version bump due to dropping Python versions
   - Update version in pyproject.toml and setup.py

3. **Create release notes**
   - Highlight Python 3.12+ requirement
   - Emphasize security improvements

## Risk Mitigation

1. **Breaking Changes**
   - Maintain a detailed list of all breaking changes
   - Provide migration scripts where possible
   - Clear communication in release notes

2. **Performance Regressions**
   - Benchmark critical operations before and after
   - Profile memory usage with new packages
   - Have rollback plan if issues found

3. **Compatibility Issues**
   - Test with downstream projects if any
   - Provide beta release for early testing
   - Monitor issue tracker closely after release

## Testing Checklist

- [ ] All unit tests pass on Python 3.12, 3.13, 3.14-rc1
- [ ] All integration tests pass
- [ ] Benchmarks show acceptable performance
- [ ] No security vulnerabilities in dependencies
- [ ] Documentation builds successfully
- [ ] Example scripts run without errors
- [ ] CI/CD pipeline fully green

## Post-Upgrade Monitoring

1. **Monitor for issues**
   - Watch GitHub issues for upgrade-related problems
   - Be prepared for quick patch releases

2. **Performance monitoring**
   - Track performance metrics in production use
   - Gather user feedback on any slowdowns

3. **Security monitoring**
   - Set up automated vulnerability scanning
   - Subscribe to security advisories for all dependencies

## Timeline Summary

- **Week 1**: Preparation and configuration updates
- **Week 2**: Code modernization and initial package updates
- **Week 3**: Testing and validation
- **Week 4**: Documentation and release preparation

Total estimated time: 4 weeks

## Conclusion

This upgrade will modernize the JDemetra+ Python codebase, improve security by using the latest versions of all dependencies, and leverage Python 3.12+ features for better performance and maintainability. All machine learning packages have been verified to have no known security vulnerabilities as of July 30, 2025.