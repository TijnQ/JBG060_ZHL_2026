# Model research results

This file records the main results of models that were tested on real
data. Detailed prediction tables are written to `modeling/outputs`.
That folder is not stored in Git because it can contain large files.

## TabPFN county-level baseline

### Goal

TabPFN was tested as a baseline for two related tasks:

- predicting whether a flood occurs
- predicting how many square kilometres are flooded

The first completed experiment used the five counties in the Aweil study
area. The national experiment has not been completed yet.

### Data split

| Part | Rows |
|---|---:|
| Training | 3,915 |
| Validation | 1,300 |
| Test | 1,560 |

The model used 50 features. The data was split by time. This prevents
future observations from being used to predict the past.

### Main test results

| Metric | TabPFN | Persistence |
|---|---:|---:|
| MAE | 3.895 km² | 2.883 km² |
| RMSE | 16.013 km² | 10.999 km² |
| Skill score | -0.456 | Not applicable |
| CRPS | 2.391 | Not calculated |
| Brier score | 0.067 | Not calculated |
| Probability of Detection | 0.789 | Not calculated |
| False Alarm Ratio | 0.123 | Not calculated |
| Critical Success Index | 0.711 | Not calculated |

Persistence predicts that the next flooded area will be equal to the
most recently observed flooded area. It is a simple baseline, but it is
strong when flooded area changes slowly.

### Flood detection

TabPFN was strong at predicting whether a flood occurred:

- ROC-AUC was 0.945
- average precision was 0.896
- about 79 percent of flood events were detected
- the Brier score was 0.067
- the Brier skill score was 0.658 compared with a constant probability
  baseline

These results show that the model can separate flood and non-flood
observations well.

### Flooded area and spike events

TabPFN was weaker at predicting the exact flooded area. Persistence had
a lower MAE and RMSE. The negative skill score also shows that TabPFN did
not beat persistence.

The weakness was clearest during 166 spike events:

| Spike metric | TabPFN | Persistence |
|---|---:|---:|
| Mean prediction | 8.558 km² | Not applicable |
| Mean real value | 40.217 km² | Not applicable |
| MAE | 32.005 km² | 18.846 km² |
| RMSE | 48.471 km² | 30.598 km² |

TabPFN often noticed that a flood was happening, but it strongly
underestimated how large the flood became.

### Conclusion

TabPFN is useful as a baseline for flood detection. It needs almost no
hyperparameter tuning and gives strong probability predictions. It is
not the best model for predicting the exact flooded area in the current
setup. It performs worse than persistence and underestimates extreme
events.

A possible next step is a two-stage model. The first model predicts
whether flooding occurs. A second model predicts the flooded area only
when a flood is expected.

### Reproduction status

- The Aweil county run was completed on 29 September 2026.
- The shared modeling checks pass.
- Feature caching is available for faster repeat runs.
- A repeated GPU run took about 35 seconds on the test computer.
- The national TabPFN run has not been completed.
- Every user needs their own Prior Labs account and accepted licence.
- API keys must not be saved in Git.
