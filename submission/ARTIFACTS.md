# Artifacts — Lab21

GitHub records CP1–CP5 results, report, reflection and source code. The FAILED regression verdict is an honest, gradeable result.

The correct adapter was created at `/content/Day21-Track3-Finetuning-Lab/adapters/correct` in the [Colab notebook](https://colab.research.google.com/drive/1ChKb3kpcQM96O7MZUjg5hvgWp5OsJdwJ).
Weights are excluded from ordinary Git under the repository artifact policy. For a ZIP submission requiring weights, download adapters/correct from Colab with results and submission. Colab storage is temporary.

Reproduce on T4 with EVAL_LIMIT unset and epochs=2:

```bash
python scripts/colab_run.py nb1 nb2 nb3 nb4 nb5
python scripts/qualitative_compare.py
python -X utf8 scripts/verify.py
```

The diagnostic script saves complete baseline/adapter predictions without changing frozen aggregate scores, verdict or eval data. Original NB5 qualitative previews are truncated and omit baseline answers. NB6 and bonus experiments are not claimed complete.
