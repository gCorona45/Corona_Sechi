# Algoritmi di Reinforcement Learning: dai Multi-Armed Bandit al Deep Q-Learning

## Descrizione del Progetto
Questo repository contiene un'implementazione modulare e strutturata di diversi algoritmi di Reinforcement Learning (RL), sviluppata per analizzare processi decisionali in condizioni di incertezza. Il progetto è suddiviso in due aree principali: la risoluzione del problema dei $k$-bracci (Multi-Armed Bandits) e l'implementazione di agenti per ambienti a stati complessi, utilizzando sia metodi tabulari che approssimazione di funzione tramite reti neurali profonde (Deep RL).

L'obiettivo è dimostrare la transizione dalla gestione del tradeoff esplorazione-sfruttamento alla risoluzione di processi decisionali di Markov (MDP) in spazi di stato ampi o continui.

## Struttura del Software
Il codice è organizzato per garantire la separazione tra la logica algoritmica, la definizione degli ambienti e la fase di analisi sperimentale.

### 1. Multi-Armed Bandits (k-Brazos)
Modulo dedicato allo studio di strategie di selezione delle azioni in scenari stazionari e non stazionari.
- **Modellazione dei Bracci**: Implementazioni basate su distribuzioni di probabilità Bernoulli, Binomiale e Normale (`src/armbernoulli.py`, `src/armbinomial.py`, `src/armnormal.py`).
- **Algoritmi**: Implementazione di strategie di esplorazione, inclusa la famiglia Upper Confidence Bound (UCB).
- **Analisi**: Notebook dedicati alla valutazione della scalabilità e del benchmark delle performance in base alla distribuzione della ricompensa.

### 2. Ambienti Complessi (Entornos Complejos)
Modulo focalizzato su agenti capaci di operare in ambienti con elevata dimensionalità.
- **Metodi Tabulari**: Implementazione di algoritmi classici per la risoluzione di MDP.
- **Metodi Approssimati**: Utilizzo di SARSA con approssimazione lineare (Semi-Gradient).
- **Deep Reinforcement Learning**: Implementazione di un'architettura Deep Q-Network (DQN).
    - `DQN_Network.py`: Definizione dell'architettura della rete neurale tramite framework di Deep Learning.
    - `DQNAgent.py`: Logica dell'agente comprendente meccanismi di Experience Replay e Target Network per la stabilizzazione dell'apprendimento.

## Specifiche Tecniche e Implementazione
- **Linguaggio**: Python 3.12+
- **Visualizzazione**: Moduli dedicati (`plotting.py`) per il monitoraggio delle curve di apprendimento e della convergenza delle policy.
- **Documentazione Scientifica**: Il repository include un report tecnico dettagliato (`informe.pdf`) che analizza i risultati sperimentali e le scelte iperparametriche.

## Requisiti e Installazione

### Prerequisiti
È necessaria un'installazione di Python (versione 3.12 o superiore).

### Setup Ambiente
1. Clonare il repository:
   ```bash
   git clone [https://github.com/gCorona45/Corona_Sechi.git](https://github.com/gCorona45/Corona_Sechi.git)
   cd Corona_Sechi
