# WMT26 Terminology Shared Task

Evaluation tooling for the [WMT26 terminology shared task](https://www2.statmt.org/wmt26/terminology.html):
document-level MT with terminology guidance for en→pl, es→eu (tracks 1 and 2) and zh→en (track 2).

- `data/public/`: the released competitor-facing data and the official validation script
- `data/unified/`: unified test sets **with gold data**

```bash
make install
make evaluate-gold                                # score the references against themselves
make evaluate SUBMISSIONS=dir OUT=results         # score {system}.{mode}.{domain}.{pair}.json files
```
