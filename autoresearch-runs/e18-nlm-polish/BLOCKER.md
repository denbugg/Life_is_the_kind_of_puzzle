# Kaggle API blocker

The E18b package is locally complete and verified.  No Kaggle API call or push
was attempted because the previously observed repeated HTTP 404 blocker has not
been declared cleared.

`push_e18b_kaggle.sh` exits with status 2 unless
`KAGGLE_404_BLOCKER_CLEARED=1` is explicitly set.  After API recovery, run:

```bash
KAGGLE_404_BLOCKER_CLEARED=1 bash push_e18b_kaggle.sh
```
