## CortexLTM
Schema-driven long-term memory layer for LLMs and agents.

Make sure you have/are on:  
`winget install -e --id Python.Python.3.12`  
64 bit not 32 bit

## Basic Project Setup

1) In your repo, create a Python venv by running -

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install the Groq SDK or similar -

```powershell
pip install groq
```

3) If using .env install -

```powershell
pip install python-dotenv
```

4) Refer to `.env.example`

5) Create `groq_test.py` (or similar) to load .env and test you get a response. Run with -

```powershell
python groq_test.py
```

6) if you have issues with unresolved imports do this -

Press Ctrl + Shift + P  
Type: Python: Select Interpreter  
Pick: `C:\myproject\.venv\Scripts\python.exe` (or whatever your project path is).

7) Install the DB driver in the same activated venv terminal -

```powershell
pip install psycopg-binary
```

## Run Scripts in sql folder

Scripts are numbered in the order they were ran. It is highly recommended to run them in the exact order as they are listed.
