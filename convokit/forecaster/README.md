**Table 1: Forecasting derailment on CGA-CMV-large conversations.**
The performance is measured in accuracy (Acc), precision (P), recall (R), F1, false positive rate (FPR), mean horizon (Mean H), and Forecast Recovery (Recovery) along with the correct and incorrect recovery rates. Results are reported as averages over five runs with different random seeds.

| Model             | Acc ↑  | P ↑     | R ↑    | F1 ↑    | FPR ↓    | Mean H ↑ | Recovery ↑ (CA/N - IA/N) |
|-------------------|--------|---------|--------|---------|----------|----------|--------------------------|
| Gemma2 9B         | $71.0$ | $69.1$  | $76.1$ | $72.3$  | $34.2$   | $3.9$   | $+1.8 (8.4 - 6.6)$       |
| Mistral 7B        | $70.7$ | $68.8$  | $76.0$ | $72.1$  | $34.6$   | $4.0$   | $+2.9 (8.1 - 5.2)$       |
| Phi4 14B          | $70.5$ | $67.7$  | $78.4$ | $72.6$  | $37.5$   | $4.0$   | $+2.0 (7.7 - 5.7)$       |
| LlaMa3.1 8B       | $70.0$ | $68.8$  | $73.2$ | $70.9$  | $33.2$   | $4.0$   | $+1.7 (7.3 - 5.6)$       |
| DeBERTaV3-large   | $68.9$ | $67.3$  | $73.7$ | $70.3$  | $36.0$   | $4.2$   | $+1.1 (7.6 - 6.5)$       |
| RoBERTa-large     | $68.6$ | $67.1$  | $73.4$ | $70.0$  | $36.1$   | $4.2$   | $+1.6 (7.5 - 5.9)$       |
| RoBERTa-base      | $68.1$ | $67.3$  | $70.6$ | $68.8$  | $34.4$   | $4.2$   | $+0.7 (7.4 - 6.7)$       |
| DeBERTaV3-base    | $67.9$ | $66.7$  | $71.4$ | $69.0$  | $35.7$   | $4.2$   | $+1.5 (7.2 - 5.7)$       |
| SpanBERT-large    | $67.0$ | $65.8$  | $70.5$ | $68.1$  | $36.6$   | $4.2$   | $+1.3 (8.3 - 7.0)$       |
| SpanBERT-base     | $66.4$ | $64.7$  | $72.0$ | $68.2$  | $39.3$   | $4.4$   | $+1.7 (9.6 - 8.0)$       |
| BERT-large        | $65.7$ | $66.0$  | $65.4$ | $65.5$  | $34.1$   | $4.2$   | $+0.4 (7.8 - 7.3)$       |
| BERT-base         | $65.3$ | $64.1$  | $70.1$ | $66.9$  | $39.5$   | $4.4$   | $+1.9 (9.7 - 7.8)$       |
| CRAFT             | $62.8$ | $59.4$  | $81.1$ | $68.5$  | $55.5$   | $4.7$    | $+4.9 (12.0 - 7.1)$      |


**Table 2: Forecasting derailment on CGA-Wikiconv conversations.**
The performance is measured in accuracy (Acc), precision (P), recall (R), F1, false positive rate (FPR), mean horizon (Mean H), and Forecast Recovery (Recovery) along with the correct and incorrect recovery rates. Results are reported as averages over five runs with different random seeds.

| Model             | Acc ↑  | P ↑     | R ↑    | F1 ↑    | FPR ↓    | Mean H ↑ | Recovery ↑ (CA/N - IA/N) |
|-------------------|--------|---------|--------|---------|----------|----------|--------------------------|
| Gemma2 9B         | $69.2$ | $67.5$  | $75.3$ | $70.9$  | $36.9$   | $3.6$    | $+0.9 (4.1 - 3.2)$       |
| Phi4 14B          | $68.8$ | $69.5$  | $67.1$ | $68.2$  | $29.6$   | $3.3$    | $+0.8 (3.7 - 2.9)$       |
| LlaMa3.1 8B       | $68.5$ | $66.3$  | $75.6$ | $70.5$  | $38.7$   | $3.6$    | $+1.8 (5.5 - 3.7)$       |
| RoBERTa-large     | $68.2$ | $67.8$  | $69.7$ | $68.6$  | $33.3$   | $3.6$    | $+0.3 (3.9 - 3.5)$       |
| SpanBERT-large    | $67.9$ | $66.5$  | $72.6$ | $69.3$  | $36.7$   | $3.6$    | $+0.1 (4.9 - 4.8)$       |
| Mistral 7B        | $67.8$ | $65.9$  | $74.4$ | $69.8$  | $38.8$   | $3.8$    | $+1.1 (5.1 - 4.0)$       |
| DeBERTaV3-large   | $67.8$ | $66.9$  | $70.9$ | $68.7$  | $35.3$   | $3.7$    | $+0.8 (3.8 - 3.0)$       |
| RoBERTa-base      | $67.6$ | $65.7$  | $73.9$ | $69.5$  | $38.6$   | $3.6$    | $+0.5 (3.4 - 2.8)$       |
| DeBERTaV3-base    | $67.5$ | $67.0$  | $69.2$ | $68.0$  | $34.3$   | $3.6$    | $+0.5 (2.7 - 2.3)$       |
| SpanBERT-base     | $66.7$ | $66.1$  | $68.7$ | $67.3$  | $35.2$   | $3.3$    | $-0.7 (4.5 - 5.2)$       |
| BERT-base         | $66.5$ | $66.5$  | $66.3$ | $66.4$  | $33.4$   | $3.6$    | $-1.6 (5.6 - 7.2)$       |
| BERT-large        | $65.7$ | $65.6$  | $67.0$ | $66.0$  | $35.6$   | $3.6$    | $+0.0 (5.6 - 5.6)$       |
| CRAFT             | $64.8$ | $63.4$  | $70.1$ | $66.5$  | $40.5$   | $3.5$    | $+0.4 (3.7 - 2.9)$       |