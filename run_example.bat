@echo off
REM Windows Command Prompt example. Set keys once per session (never commit them):
REM   set GEMINI_API_KEY=your_key
REM   set OPENAI_API_KEY=your_key
REM   set ANTHROPIC_API_KEY=your_key

REM 1) Offline pipeline check (no API calls)
python -m cryptoharm run --models mock:oracle mock:erode --instances 3 --samples 1 --out results\smoke
python -m cryptoharm analyze --raw results\smoke\raw.jsonl --out results\smoke\analysis

REM 2) Real run (edit model names to ones you have access to; add more models for the paper)
REM python -m cryptoharm run --models gemini:gemini-2.5-flash --instances 10 --samples 3 --workers 4 --sleep 1 --out results\main
REM python -m cryptoharm analyze --raw results\main\raw.jsonl --out results\main\analysis
