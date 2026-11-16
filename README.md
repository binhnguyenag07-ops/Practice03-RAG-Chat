# Practice 03 — Lumen Travel RAG Chat

> **IMPORTANT:** Complete student information and the video link before submission. Review and rewrite the travel entries in your own words. The six test results below were measured against the local Ollama server on 16 November 2026.

## Student information
- Student name: **[NGUYEN CAO AN BINH]**
- Student ID: **[25127183]**
- Course: Computational Thinking — Practice 03
- Chosen city: **Ho Chi Minh City, Viet Nam**

## Features
- Email/password sign-up and sign-in using Firebase Authentication
- Protected Streamlit RAG chat and sign-out
- 15 custom travel guide entries (`knowledge_base.json`)
- `embeddinggemma` embeddings and cosine-similarity top-4 retrieval
- `qwen3:4b` responses constrained to the retrieved guide
- Retrieved IDs/scores shown in an expandable panel
- Conversation history via `st.session_state` only; no Firestore/database
- Friendly errors for invalid login and unavailable Ollama server

## Installation
1. Create a Firebase project. In Authentication > Sign-in method, enable **Email/Password**.
2. Copy the Web API Key from Firebase Project Settings.
3. Set up Ollama with `embeddinggemma` and `qwen3:4b`. Use the Lab 4 Pinggy tunnel (or follow its instructions).
4. Create a local `.env` from `.env.example`; enter the Firebase key and the HTTPS Pinggy URL.
5. In PowerShell, inside the project directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

If using a locally running Ollama instance to debug, `OLLAMA_URL=http://localhost:11434` may be used temporarily. The assignment requires the Ollama + Pinggy server from Lab 4 for the submitted solution.

### Start the Pinggy tunnel
Keep Ollama running. Download the Windows CLI from https://pinggy.io/cli/ and place `pinggy.exe` in the project folder (the binary is excluded from Git). Open PowerShell here and run:
```powershell
.\pinggy.exe -l http://127.0.0.1:11434 'u:Host:localhost:11434' --b
```
Copy the HTTPS URL printed by Pinggy into `OLLAMA_URL` in `.env` (base URL only, without `/api`). The tunnel runs in the background. Refresh Streamlit and submit a new question so the app reloads `.env`. Use `.\pinggy.exe ps` to inspect running tunnels and `.\pinggy.exe stop TUNNEL_ID` to stop this tunnel after the demo. When creating a new tunnel, update `.env` if its URL changes.

### Required Ollama models
```powershell
ollama pull embeddinggemma
ollama pull qwen3:4b
ollama list
```

### Test server access
```powershell
Invoke-RestMethod http://localhost:11434/api/tags
# Replace URL with actual Pinggy tunnel URL for tunnel verification:
Invoke-RestMethod https://YOUR-TUNNEL.pinggy.link/api/tags
```

## System flow
`Firebase sign-in → embed question → cosine similarity → top-k retrieval → build context → qwen3:4b generation → display answer + source IDs/scores`

### Timeout fix and verification
The installed `qwen3:4b` template always opened `<think>` even when `think=false`; the server log showed approximately 972 generated tokens before a request was canceled at 180 seconds. The model runs on CPU (`size_vram=0`). The app now requests a JSON schema with a short final `answer` and `source_ids`, limits generation to 192 tokens and the context to 2048 tokens, and reads streamed NDJSON. It validates cited IDs against the retrieved entries and standardizes responses with no supporting IDs to the required unknown answer. HTTP errors identify the failing stage. The guide index remains cached.

The six live tests used `http://127.0.0.1:11434`, `embeddinggemma`, `qwen3:4b`, k=4 and threshold 0.30. Timings were 5.17–19.84 seconds per question; this is a measurement, not a latency guarantee. Raw answers and retrieved IDs are in `rag_test_results.json`. Rerun through the Pinggy URL before recording the submission demo.

An additional Streamlit AppTest using the current `.env` and real Ollama returned the correct Post Office answer with [G03] in 48.64 seconds including app initialization. Six offline regression tests passed. Separate UI tests with a simulated signed-in session checked the login gate, message history and sign-out clearing; these did not create a real Firebase account. The existing Streamlit server health check returned `ok`.

Offline regression checks:
```powershell
python -m unittest -v test_rag
```

## Guide schema
Each JSON object has `id`, `title`, `text`, `source`, plus `category` and `tags` for the following assignment RAG-Rec. Entries deliberately omit volatile operating hours and entry prices, so questions about such details must return **I don't know based on the travel guide.**

## Test table (actual local Ollama results, 16 November 2026)
| Type | Question | Expected ID | Retrieved IDs (actual) | Correct? (actual) | Failure stage if incorrect |
|---|---|---|---|---|---|
| Answerable | What is special about the Saigon Central Post Office? | G03 | G03, G09, G04, G07 | Yes: grand hall, yellow exterior, historic architecture [G03]; 19.84 s | None |
| Answerable | What can visitors buy at Ben Thanh Market? | G05 | G05, G12, G13, G09 | Yes: local goods, souvenirs, food [G05]; 13.73 s | None |
| Answerable | Which church has a pink exterior? | G11 | G11, G04, G03, G10 | Yes: Tan Dinh Church [G11]; 12.91 s | None |
| Paraphrase | Where can I walk along a broad pedestrian boulevard in the city center? | G06 | G06, G09, G05, G13 | Yes: Nguyen Hue Walking Street [G06]; 12.38 s | None |
| Out of guide | How much is the entrance ticket to the War Remnants Museum? | None — must say I don't know | G02, G08, G01, G05 | Yes: exact unknown answer; 7.58 s | None |
| Out of guide | What time does Ben Thanh Market close today? | None — must say I don't know | G05, G12, G13, G09 | Yes: exact unknown answer; 5.17 s | None |

**Evaluation rule:** Mark a question correct only after checking actual app output, whether the answer is supported by the relevant guide entry, and whether cited IDs match evidence. The last two rows should return an explicit unknown answer, even if relevant entries are retrieved.

## Computational Thinking
- **Abstraction:** Input is an authenticated user and natural-language question. Outputs are an answer, cited guide IDs and retrieved scores. Guide entries keep `id`, `title`, `text`, `source`, `category`, and `tags` for traceable answers and future recommendation tasks.
- **Decomposition:** Sign in → embed → retrieve → build context → generate → display. Each stage has clear data inputs and outputs.
- **Pattern recognition:** Reuse Firebase Authentication/Streamlit from Lab 3, Ollama/Pinggy and semantic retrieval from Lab 4. The recurring pattern is filter (consider the guide) → score (cosine) → top-k (highest-scoring entries), reused in RAG-Rec for ranking places.
- **Algorithm design:** Embed the guide once; on each question, embed the question, calculate cosine similarities, select top-k, use retrieved entries to build the prompt, and generate or refuse unsupported answers. Chosen `k=4` balances enough context with reduced irrelevant material. `MIN_SIMILARITY=0.30` is an initial heuristic; evaluate and adjust it using measured cases.

## Known limitations
- A small guide may omit facts that users care about. No live opening hours or fees are provided.
- Similarity scores are not probabilities. The initial threshold needs calibration using actual tests.
- The LLM may still produce unsupported text or incorrect citations; verify its answers against the guide.
- Session-only login state and token expiry handling are simplified for this course prototype; no server-side session persistence.
- Pinggy free tunnel URLs may change and the server must remain running during use.

## AI-tool usage disclosure
OpenAI Codex assisted with source code review, diagnosing and fixing the Ollama timeout, structured answer generation and citation validation, regression and live RAG testing, Pinggy configuration, and README documentation. The student should describe their own review and modifications when presenting the work.

## Demonstration video (maximum 5 minutes)
**[[ADD YOUR OWN PUBLIC VIEWABLE YOUTUBE/GOOGLE DRIVE VIDEO LINK](https://drive.google.com/drive/u/0/folders/17LNdfmXivdb72jUzQzEindhdvq1u8i95)]**

## Repository
**[[ADD YOUR PUBLIC GITHUB REPOSITORY LINK](https://github.com/binhnguyenag07-ops/Practice03-RAG-Chat)]**
