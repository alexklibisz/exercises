
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

- TODO:
  - Could probably improve accuracy by tuning the GradientBoostingClassifier parameters
