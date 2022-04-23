from typing import Dict, Type, Union

from nba_betting.model.classes.linear import (
    DefenceLogReg,
    EfficiencyLogReg,
    OffenceLogReg,
)

NBAModel = Union[
    DefenceLogReg,
    EfficiencyLogReg,
    OffenceLogReg,
]

NBA_MODELS: Dict[str, Type[NBAModel]] = {
    DefenceLogReg.model_name: DefenceLogReg,
    OffenceLogReg.model_name: OffenceLogReg,
    EfficiencyLogReg.model_name: EfficiencyLogReg,
}
