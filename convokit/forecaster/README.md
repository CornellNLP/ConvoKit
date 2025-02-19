**Table 1: Forecasting derailment on \newcmv conversations.**  
The performance is measured in accuracy (Acc), precision (P), recall (R), F1, false positive rate (FPR), mean horizon (Mean H), and Forecast Recovery (Recovery) along with the correct and incorrect adjustment rates. The best performance across each metric is indicated in **bold**.
| Model                          | Acc ↑  | P ↑   | R ↑   | F1 ↑  | FPR ↓  | Mean H ↑ | Recovery ↑ (CA/N - IA/N) |
|--------------------------------|--------|-------|-------|-------|--------|---------|-------------------------|
| Human (84 convos) round-1      | 62.2   | 67.8  | 48.9  | 54.6  | 24.4   | 3.64    | -                     |
| Human (84 convos) round-2      | 70.0   | 75.9  | 55.6  | 63.9  | 15.6   | 3.13    | -                     |
|--------------------------------|--------|-------|-------|-------|--------|---------|-------------------------|
| RoBERTa-large                  | **68.4** | 67.5  | 71.1  | 69.2  | 34.3   | 4.14    | +1.1 (7.2 - 6.1)        |
| Gemma-2 27B-IT (finetuned)     | **68.4** | 66.2  | 75.2  | **70.4** | 38.5   | 4.30    | +0.0 (10.7 - 10.7)     |
| GPT-4o (12/2024; zero-shot)    | 66.6   | **71.0** | 56.3  | 62.8  | **23.0** | 3.78    | -1.5 (5.9 - 7.4)       |
| BERT-base                      | 65.2   | 63.5  | 72.0  | 67.4  | 41.6   | 4.45    | +2.1 (9.8 - 7.7)        |
| CRAFT                          | 62.8   | 59.4  | 81.1  | 68.5  | 55.5   | 4.69    | +4.9 (12.0 - 7.1)       |
| Gemma-2 27B-IT (zero-shot)     | 59.4   | 55.7  | **92.2** | 69.4  | 73.5   | **5.27** | **+7.1** (12.2 - 5.1)  |

