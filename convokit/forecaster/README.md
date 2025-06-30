**Table 1: Forecasting derailment on CGA-CMV-large conversations.**
The performance is measured in accuracy (Acc), precision (P), recall (R), F1, false positive rate (FPR), mean horizon (Mean H), and Forecast Recovery (Recovery) along with the correct and incorrect recovery rates. Results are reported as averages over five runs with
different random seeds.

| Model             | Acc ↑  | P ↑     | R ↑    | F1 ↑    | FPR ↓    | Mean H ↑ | Recovery ↑ (CA/N - IA/N) |
|-------------------|--------|---------|--------|---------|----------|----------|--------------------------|
| Gemma2 9B         | $71.0$ | $69.1$  | $76.1$ | $72.3$  | $34.2$   | $3.91$   | $+1.8 (8.4 - 6.6)$       |
| Mistral 7B        | $70.7$ | $68.8$  | $76.0$ | $72.1$  | $34.6$   | $3.98$   | $+2.9 (8.1 - 5.2)$       |
| Phi4 14B          | $70.5$ | $67.7$  | $78.4$ | $72.6$  | $37.5$   | $4.08$   | $+2.0 (7.7 - 5.7)$       |
| LlaMa3.1 8B       | $70.0$ | $68.8$  | $73.2$ | $70.9$  | $33.2$   | $3.95$   | $+1.7 (7.3 - 5.6)$       |
| DeBERTaV3-large   | $68.9$ | $67.3$  | $73.7$ | $70.3$  | $36.0$   | $4.19$   | $+1.1 (7.6 - 6.5)$       |
| RoBERTa-large     | $68.6$ | $67.1$  | $73.4$ | $70.0$  | $36.1$   | $4.16$   | $+1.6 (7.5 - 5.9)$       |
| RoBERTa-base      | $68.1$ | $67.3$  | $70.6$ | $68.8$  | $34.4$   | $4.15$   | $+0.7 (7.4 - 6.7)$       |
| DeBERTaV3-base    | $67.9$ | $66.7$  | $71.4$ | $69.0$  | $35.7$   | $4.24$   | $+1.5 (7.2 - 5.7)$       |
| SpanBERT-large    | $67.0$ | $65.8$  | $70.5$ | $68.1$  | $36.6$   | $4.23$   | $+1.3 (8.3 - 7.0)$       |
| SpanBERT-base     | $66.4$ | $64.7$  | $72.0$ | $68.2$  | $39.3$   | $4.36$   | $+1.7 (9.6 - 8.0)$       |
| BERT-large        | $65.7$ | $66.0$  | $65.4$ | $65.5$  | $34.1$   | $4.15$   | $+0.4 (7.8 - 7.3)$       |
| BERT-base         | $65.3$ | $64.1$  | $70.1$ | $66.9$  | $39.5$   | $4.41$   | $+1.9 (9.7 - 7.8)$       |
| CRAFT             | $62.8$ | $59.4$  | $81.1$ | $68.5$  | $55.5$   | $4.7$    | $+4.9 (12.0 - 7.1)$      |
