## GenAI Proof of Concept: perform major/minor Python version upgrade for a complex Project
The purpose of this proof of concept is to find out if an LLM can take an existing complex Python codebase and upgrade it from a lower major/minor version of Python to a higher major/minor version. For our PoC, it will be from Python 3.8 to 3.12. The challenging aspect of this migration will be migration of related machine learning libraries, while taken into account of security vulnerabilities (i.e., all upgraded packages must not have security vulnerabilities). The project we will be using for our PoC is the JDemetra+ (toolkit for seasonal adjustment and time series analysis) Python version: https://github.com/thuduc/jdplus-main-java2python

### LLM & AI Tool
* LLM used: Claude Opus 4 (best coding LLM) - https://www.anthropic.com/claude/opus
* AI tool used: Claude Code (best coding CLI due to its integration with Clause 4 LLMs) - https://www.anthropic.com/claude-code

### Conversion Process: 
* Step 1 - use Claude Code (together with Opus 4 LLM) to analyze an existing project's codebase, then ask it to put together a comprehensive upgrade plan for the Python version upgrade.
* Step 2 - developer verifies the upgrade plan and modifies the plan as needed. Developer could use Claude Code and iterate through this process until the plan is ready.
* Step 3 - use this upgrade plan (see [PYTHON_VERSION_UPGRADE.md](PYTHON_VERSION_UPGRADE.md)) in Claude Code (together with Claude Opus 4 LLM) to implement all phases in the plan.

### PoC Results
* The upgrade plan estimated a timeline of 4 weeks to complete. It took Claude Code about 2 hours to complete.
* Successful passing of all unit and integration tests. See [TEST_SUMMARY.md](TEST_SUMMARY.md) for details.

## Running the code
See jdemetra_py/README.md

## All prompts issued to Claude Code
The complete list of prompts issued to Clause Code is listed below:

> Think hard to create a plan to upgrade this project to support python 3.12 at the minimum and save this plan to PYTHON_VERSION_UPGRADE.md. Prior versions of Python will not be supported. The plan also needs to include upgrading all related machine learning packages to their latest version. Make sure none of these machine learning packages have reported security vulnerablities

> Go ahead and implement all tasks in this plan.

> Run all tests and make sure there are no test failures. Store the test results in TEST_SUMMARY.md
