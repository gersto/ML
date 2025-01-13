### Links

Das beste Opensource LLM: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard

ChatBot Arena: https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard


Huggingface: https://huggingface.co/

Github: https://github.com/

Google Colab: https://colab.research.google.com/

Installation Node: https://nodejs.org/en

Installation Ollama: https://ollama.com/

Installation LM Studio: https://lmstudio.ai/

Anything LLM: https://useanything.com/

https://github.com/Mintplex-Labs/anything-llm/blob/master/README.md


LAAMA: https://ai.meta.com/llama/


Opensource mit Interface:

https://huggingface.co/chat/

https://groq.com/


Token:

https://platform.openai.com/tokenizer

https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them


Eine wichtige Lernmethode:

RLHF: https://huggingface.co/blog/rlhf


Prompting:

https://www.promptingguide.ai/techniques/tot

https://learnprompting.org/docs/intro


RAG:

https://aws.amazon.com/de/what-is/retrieval-augmented-generation/

https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/

https://research.ibm.com/blog/retrieval-augmented-generation-RAG

https://www.databricks.com/glossary/retrieval-augmented-generation-rag

PDFs vorbereiten für RAG: https://github.com/run-llama/llama_parse

Colab Notebook (Llama_parse): https://colab.research.google.com/drive/1P-XpCEt4QaLN7PQk-d1irliWBsVYMQl5?usp=sharing

Webseiten vorbereiten für RAG: https://www.firecrawl.dev/


KI-Agenten

https://botpress.com/blog/what-is-an-ai-agent

https://voyager.minedojo.org/

https://flowiseai.com/

Flowise auf Github: https://github.com/FlowiseAI/Flowise


TTS Colab: https://colab.research.google.com/drive/17xcyh-mFWye30WwNl7wIce1kzBFNMbcQ

Finetuning in Colab: https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing#scrollTo=FqfebeAdT073


Papers:
https://arxiv.org/pdf/2307.02483

https://arxiv.org/pdf/2307.15043

https://arxiv.org/pdf/2306.13213

https://arxiv.org/pdf/2302.12173

https://arxiv.org/pdf/2305.00944


https://embracethered.com/blog/posts/2023/google-bard-data-exfiltration/

## Voraussetzungen

- Cloud, Lokal, Ausborgen von GPU-Power
- GPU (Grafikkarte)
  - NVIDIA-GPU
    - am besten GPUs mit hoher Anzahl an CUDA-Kernen und großen VRAM (Video-RAM)
    - NVIDIA RTX 3090 oder 4090 (24GB VRAM), 4080 ist auch OK
    - NVIDIA A100 oder V100 (40-80GB VRAM)
    - NVIDIA H100
    - für kleiner Modelle RTX 2080 oder 3080 (10-12GB VRAM)
- CPU (Prozessor)
  - ein leistungsstarker Mehrkernprozessor ist wichtig
    - empfohlen: Intel Core i7/i9 ider AMD Ryzen 7/9 Serien
    - für höchste Leistung: AMD Threadripper oder Intel Xeon
- RAM (Arbeitsspeicher)
  - der RAM-Bedarf variiert je nach Modell und Aufgabe
    - mindestens 32GB RAM, optimal 64GB oder mehr
- Speicher
  - schneller und ausreichend großer Speicher
    - NVMe-SSD mit mindestens 1TB Speicherplatz
- Software-Umgebung
  - Betriebssystem: Linux (z.B. Ubuntu wird häufig bevorzugt, da es bessere
    Unterstützung für viele ML-Bibliotheken bietet
  - CUDA und cuDNN: für NVIDIA GPUs werden CUDA und cuDNN Bibliotheken benötigt
  - Python
  - Deep Learning Frameworks: PyTorch oder TensorFlow
- Kühlung und Stromversorgung

- bei nicht so starker Hardware --> Quantifizierung

## Quantifizierung



## LM Studio

- downloads von Huggingface
- downloads innerhalb von Anwendungen (LM-Studio, Olama, ...))
- Modelle können auch in Notebooks laufen, die beste Wahl ist aber LM-Studio
- alle Modelle von Huggingface (https://huggingface.co/models) findest du auch in LM-Studio
- Dokumentation unter https://lmstudio.ai/docs

### Installation
- über den Link https://lmstudio.ai/
- Download für das jeweilige Betriebssystem und Installation

### Beschreibung

- Suchen nach Modellen (Mistral-7b, crok, ...)
- auf der rechten Seite kann man dann noch Untermodelle auswählen
- es wird auch angezeigt ob diese Modelle für den jeweiligen Rechner geeignet sind
- über download das jeweilige Modell herunterladen


## Zensierte vs Unzensierte LLMs

- Closed Source Modelle (Chatgpt, ...) sind natürlich zensiert
- Sie haben einen Bias (politisch, gegen gewisse Menschen, ...)
- auch das LLama_Modell (Facebook), obwohl OpenSource hat einen Bias
- durch Feintuning kommt meist noch mehr Bias (z.B. Google soll Bilder von Soldaten aus dem 2.Weltkrieg erstellen
  --> diese Bilder zeigten schwarze und japanische Personen)
- teste z.B. ein llama3-Modell mit der Frage "Wie man in ein Auto einbricht"
- wie bekommt man Modelle ohne Bias (Seite von Eric Hartford --> Dolphin- und Samantha-Modelle)
- Fragen: mache einen Witz über Frauen; Zeige mir eine Backdoor-Attacke auf das Windows-Betriebssystem

## LLMs Anwendungen

- Texterstellung und -bearbeitung
- Programmierunterstützung
- Sprachübersetzung
- Kundenunterstützung (Chatbots, ...)
- Datenanalyse


## Alternative Methoden zum Betrieb von LLMs