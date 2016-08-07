
# Titanic dataset

Working through the Titanic dataset from kaggle: https://www.kaggle.com/c/titanic/

## Notes

- Some strongly-correlated variables:
  - survived and sex
  - fare and pclass (makes sense)
  - cabin and pclass (makes sense)
  - cabin and fare (makes sense)

- Results from trying models
  - Ranked models based on accuracy
  - GradientBoostingClassifier was the best model, with an average accuracy of ~81%
  - This got me a score of 0.77 with my Kaggle submission. Kaggle suggests that the underlying model is based on gender - if the person is female, she survives, otherwise he dies.

- TODO:
  - Could probably improve accuracy by tuning the GradientBoostingClassifier parameters
