# Proof-of-concept: Python major version upgrade using Claude Code
JDemetra+ (toolkit for seasonal adjustment and time series analysis) Python source repo is located at https://github.com/thuduc/jdplus-main-java2python

This POC is to evaluate Claude Code (an agentic coding tool from Anthropic: https://www.anthropic.com/claude-code) for its ability to upgrade an existing project codebase from Python 3.8 to Python 3.12, including upgrading all related machine learning packages to their latest version. All upgraded packages must not have security vulnerabilities.

#### Conversion Process: 
* Step 1 - use a reasoning LLM that's able to analyze an existing code repository, then put together a comprehensive upgrade plan for upgrading the entire project's codebase from Python 3.8 to 3.12. We used Anthropic's Claude Opus 4 LLM for our reasoning LLM. We chose Opus 4 over OpenAI's ChatGPT o3 (advanded reasoning) and Google Gemini 2.5 Pro (reasoning) due to its advanced ability to analyze code.
* Step 2 - developer verifies and modifies this plan (python_major_version_upgrade.md) if needed.
* Step 3 - use this upgrade plan (see python_major_version_upgrade.md) with Claude Code (together with Claude Opus 4 LLM, known as the most advanded model for agentic coding tasks) to implement all tasks in all phases defined in the upgrade plan. The upgrade plan includes requirements for comprehensive test coverage.

The Python major version upgrade took Claude Code about 2 hours to complete. This includes the successful passing of all unit and integration tests. See jdemetra_py/TEST_SUMMARY.md for details. The converted python codebase resides under jdemetra_py folder.


## Running the code
See jdemetra_py/README.md

## All prompts issued to Claude Code
The complete list of prompts issued to Clause Code is listed below:

> Think hard to create a plan to upgrade this project to support python 3.12 at the minimum and save this plan to python_major_version_upgrade.md. Prior versions of Python will not be supported. The plan also needs to include upgrading all related machine learning packages to their latest version. Make sure none of these machine learning packages have reported security vulnerablities

> Go ahead and implement all tasks in this plan.

> Run all tests and make sure there are no test failures. Store the test results in TEST_SUMMARY.md
