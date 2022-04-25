from typing import Dict, Type, Union

from nba_betting.model.classes.linear import (
    DefenceLogReg,
    EfficiencyLogReg,
    OffenceLogReg
)

from nba_betting.model.classes.naive_bayes import (
    DefenceGaussianNB,
    OffenceGaussianNB,
    EfficiencyGaussianNB
)

from nba_betting.model.classes.random_forest import (
    DefenceRandForest,
    OffenceRandForest,
    EfficiencyRandForest
)

NBAModel = Union[
    DefenceLogReg,
    EfficiencyLogReg,
    OffenceLogReg,
    DefenceGaussianNB,
    OffenceGaussianNB,
    EfficiencyGaussianNB,
    DefenceRandForest,
    OffenceRandForest,
    EfficiencyRandForest
]

NBA_MODELS: Dict[str, Type[NBAModel]] = {
    DefenceLogReg.model_name: DefenceLogReg,
    OffenceLogReg.model_name: OffenceLogReg,
    EfficiencyLogReg.model_name: EfficiencyLogReg,
    DefenceGaussianNB.model_name: DefenceGaussianNB,
    OffenceGaussianNB.model_name: OffenceGaussianNB,
    EfficiencyGaussianNB.model_name: EfficiencyGaussianNB,
    DefenceRandForest.model_name: DefenceRandForest,
    OffenceRandForest.model_name: OffenceRandForest,
    EfficiencyRandForest.model_name: EfficiencyRandForest,
}
