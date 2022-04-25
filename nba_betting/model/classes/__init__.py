from typing import Dict, Type, Union

from nba_betting.model.classes.linear import (
    DefenceLogReg,
    EfficiencyLogReg,
    GeneralLogReg,
    OffenceLogReg,
    ShootingLogReg,
)
from nba_betting.model.classes.naive_bayes import (
    DefenceGaussianNB,
    EfficiencyGaussianNB,
    GeneralGaussianNB,
    OffenceGaussianNB,
)
from nba_betting.model.classes.random_forest import (
    DefenceRandForest,
    EfficiencyRandForest,
    OffenceRandForest,
)

NBAModel = Union[
    DefenceLogReg,
    EfficiencyLogReg,
    OffenceLogReg,
    ShootingLogReg,
    DefenceGaussianNB,
    OffenceGaussianNB,
    EfficiencyGaussianNB,
    DefenceRandForest,
    OffenceRandForest,
    EfficiencyRandForest,
    GeneralLogReg,
    GeneralGaussianNB,
]

NBA_MODELS: Dict[str, Type[NBAModel]] = {
    DefenceLogReg.model_name: DefenceLogReg,
    OffenceLogReg.model_name: OffenceLogReg,
    EfficiencyLogReg.model_name: EfficiencyLogReg,
    GeneralLogReg.model_name: GeneralLogReg,
    ShootingLogReg.model_name: ShootingLogReg,
    DefenceGaussianNB.model_name: DefenceGaussianNB,
    OffenceGaussianNB.model_name: OffenceGaussianNB,
    EfficiencyGaussianNB.model_name: EfficiencyGaussianNB,
    GeneralGaussianNB.model_name: GeneralGaussianNB,
    DefenceRandForest.model_name: DefenceRandForest,
    OffenceRandForest.model_name: OffenceRandForest,
    EfficiencyRandForest.model_name: EfficiencyRandForest,
}
