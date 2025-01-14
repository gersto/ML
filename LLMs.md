## Was sind LLMs

Ein Large Language Model (LLM) ist ein Deep-Learning-Algorithmus, der enorm große Datensätze verwendet.
LLMs werden häufig in Anwendungen im Bereich des Natural Language Processing (NLP) eingesetzt,
wenn es darum geht, Abfragen in natürlicher Sprache einzugeben, um eine Antwort bzw. ein Ergebnis zu bekommen.

Ein LLM kann neue Inhalte verstehen, zusammenfassen, generieren und vorhersagen. Es verfügt typischerweise
über Milliarden von Parametern, die Erinnerungen ähneln, die das Modell während des Lernens durch Training
sammelt. Parameter ist dabei ein Begriff aus dem Bereich Machine Learning (ML). Dabei handelt es sich um
Variablen in einem Modell, an denen es trainiert wurde und die dazu dienen, neue Inhalte abzuleiten.


- besteht meistens aus 2 Files (Paramterfile und Runfile)
- Runfile aus ungefähr 500 Zeilen Code (meist Python oder C)
- Parameterfile ist eine Zusammenfassung aus Text
- z.B. Llama2 aus ungefähr 10TB Text wird ein 140GB komprimiertes File erzeugt
- daraus entsteht das Llama2 70B Parameter Modell
- dies funktioniert nur durch den Einsatz von GPU-Power
- bei einem Open-Source Modell kann man sich dies downloaden und lokal laufen lassen
- Closed-Source LLMs sind nur in einem Web-Inferface verwendbar
- generelles Konzept: Man kann eine Frage stellen und das LLM errechnet aus Wahrscheinlichkeiten eine "sinnvolle" Anwort
- die Erstellung dieser LLM Files nennt man Pretraining
- meist kommt danach, dass durch Menschen überwachte Feintuning (Fragen und Antworten werden vorgegeben und in das Modell integriert)
- danach kommt noch sehr oft das Reinforcement Learning ("Mensch gibt Daumen hoch oder hinunter" und dadurch lernt das Modell - Belohnungen)
- alle Maschinen arbeiten mit sogenannten Tokens (Zerteilung von Texten in sogenannte Tokens)
- Tokens:
  - 1 token ~ 4 chars in Englisch
  - 1 token ~ 3/4 word
  - 100 tokens ~ 75 words
  - 1-2 snetence ~ 30 tokens
  - 1 paragraph ~ 100 tokens
  - 1500 words ~ 2048 tokens
  - ansehen z.B. über einen Tokenizer (z.B. https://platform.openai.com/tokenizer) - über die TokenID sieht man wie diese Tokens wirklich gespeichert werden.
  - mit diesen Zahlen kann das neuronale Netz im Hintergrubd rechnen
- Tokenlimit --> alle Modelle haben ein Limit der Anzahl der Token (2000, 120k 1m, ...)
- jedes Modell kann sich nur eine bestimmte Anzahl von Tokens merken (OpenSource-Modelle oft nur 4000 Tokens)
- Google Gemini hat z.B. ein Limit von 1m Token

## Welche LLMs gibt es?

- OpenSource LLM
- ClosedSource LLM
- Rangliste über Leaderboards (https://lmarena.ai/?leaderboard oder https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/)
- 


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

- Grundidee: Anstatt die Zahlen in einem Modell mit hoher Genauigkeit (z.B. 32-bit) zu speichern
  und zu verarbeiten, werden sie auf eine geringere Genauigkeit (z.B. 8-bit oder 4-bit) reduziert
- Beispiel: Eine Zahl, die normalerweise als 32-bit-Gleitkommanzahl gespeichert wird, könnte als
  8-bit oder 4-bit Integer gespeichert werden

Vorteile:
- Speicherplatz sparen
- schnellere Berechnungen
- Q8: hier werden die Zahlen im Modell auf 8-bit Genauigkeit reduziert. Dies ist eine moderate
  Reduktion und bietet einen guten Kompromiss zwischen Speicherplatz, Rechenleistung und
  Modellgenauigkeit
- Q4: hier werden die Zahlen im Modell auf 4-bit Genauigkeit reduziert.
- durch Quantifuzierung können auch komplexe Modell auf weniger leistungssstarker
  Hardware ausgeführt werden.
- man kann sich dies wie bei Videoauflösungen vorstellen



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

## Feintuning eines LLM-Modells

- Voraussetzung ist ein Base-Modell (OpenSource) und danach muss man dieses feintunen
- man kann auch über die OpenAI-API feintunen - diese Modelle werden aber nur minimal angepasst
- Feintuning mit Huggingface --> https://huggingface.co/autotrain
  - Create new project
  - ...
  - kostet etwas
- Feintuning über Google Colab
  - kostet Zeit, ist aber oft kein Geld
 
## Bilderkennung mit OpenSource LLMs

- z.B. mit Llama3 oder LLava oder Phi3

## Mehr Details zur Hardware

- GPU Offload
  - je höher desto mehr wird die GPU belastet, je kleiner desto weniger
  - ohne GPU-Offloading
    - CPU führt alle Berechnungen aus
    - RAM speichert Modell und Daten
    - Speicher lagert Modell vor dem Laden
  - teilweises GPU-Offlaoding
    - CPU weniger Berechungen
    - GPU spezifische Berechungen
    - RAM etwas entlastet
    - VRAM speichert Modell und Daten
  - erhöhtes CPU-Offloading
  - maximales GPU-Offloading
    - CPU minimal belastet
    - GPU führt nahezu alle Berechnungen durch
    - RAM stark entlastet
    - VRAM stark belastet


## Alternative Methoden zum Betrieb von LLMs

### Olama

### Direkt vom Hersteller der OpenSource LLMs

#### Cohere
direkt über den Link https://cohere.com und dort im Chat direkt ddie Modelle testen

#### Lama über Meta direkt

#### Chatbot Arena (https://lmarena.ai/)

#### HuggingChat (ein Interface für die Nutzung von Open-Source LLMs)

### Prompt Engineering

Testen am Beispiel von HuggingChat (https://huggingface.co/chat/)

z.B. Code a snake game

#### Systemprompt

- z.B. You are a helpful Assistant
- You are a python-expert --> wenn du python-code schreiben lassen willst
- du bist ein kreativer Textexperte
- Du bist ein nützlicher Gehilfe und Experte für Code

- Das Standard-LLM kann normalerweise nur Text erweitern und Text zusammenfassen
- Falls man mehr will muss man den System-Prompt erweitern
- funktioniert nach dem Konzept der semantic association
- es geht darum Kontext zu geben (neuere Modelle - z.B. Copilot - fragen nach einem entsprechendem Kontext,
  wenn man diesen nicht angibt)

### Function Calling
- viele Modelle haben zusätzliche Tools integriert, welche Sie bei Bedarf aufrufen
- z.B Calculator beim Aufruf von mathematischen Aufgaben (z.B. Was ist 88 . 88 / 2)
- LLMs errechnen nur anhand von Wahrscheinlichkeiten den nächsten Token (Text, Code, ...)
- manche LLMs können zusätzliche Anfragen an andere Modelle (Fusion-Modelle, Musikerstellung, Internetanfragen, ...)

![grafik](https://github.com/user-attachments/assets/f01ca733-4c4b-4f06-a32d-20b9b79875f1)

- LLMs können als "Betriebssystem" angesehen werden und das "Betriebssystem macht Function-Calls an andere Tools
- in LM-Studio ist das Function-Calling nicht integriert --> muss selbst eingefügt werden
- ChatGPT und Hugging-Chat hat dieses Function-Calling schon eingebaut


### Vektordatenbanken und Embeddings

- das Hochladen und speichern von Daten funktioniert durch die sogenannte RAG-Technologie
- die RAG-Technologie (Retrieval-Augmented Generation) funktioniert dadurch, dass wir unser Wissen in eine sogenannte Vektordatenbank speichern
- das LLM kann dan diese Vektrodatenbank zusätzlich durchsuchen

![grafik](https://github.com/user-attachments/assets/28dd627e-5a0d-44b8-b1ad-21a9e4ab4e80)

![grafik](https://github.com/user-attachments/assets/33a1c542-33ec-4faa-bc41-e0fab9997a00)


- die embedding-Modelle machen aus den hochgeladenen Daten sogeannte Tokens, welche in einer VektorDB gespeichert werden
- in einer VektorDB werden die Daten in sogenannten Clustern gespeichert
- RAG-Technologie ist Function-Calling mit Datensuche in einer VektorDB


### Lokaler RAG-Chatbot mit Anything-LLM und LM-Studio

### Assistenten

man kann z.B. in Hugging-Chat eigene Assistenten erzeugen (oft nur Standardmodelle mit eingestelltem Systemprompt

## Links

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
