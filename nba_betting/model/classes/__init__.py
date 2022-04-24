from typing import Dict, Type, Union

from nba_betting.model.classes.linear import (
    DefenceLogReg,
    EfficiencyLogReg,
    OffenceLogReg,
)

from nba_betting.model.classes.nearest_neighbors import (
    DefenceKNeighbors,
    OffenceKNeighbors,
    EfficiencyKNeighbors,
)

NBAModel = Union[
    DefenceLogReg,
    EfficiencyLogReg,
    OffenceLogReg,
    DefenceKNeighbors,
    OffenceKNeighbors,
    EfficiencyKNeighbors,
]

NBA_MODELS: Dict[str, Type[NBAModel]] = {
    DefenceLogReg.model_name: DefenceLogReg,
    OffenceLogReg.model_name: OffenceLogReg,
    EfficiencyLogReg.model_name: EfficiencyLogReg,
    DefenceKNeighbors.model_name: DefenceKNeighbors,
    OffenceKNeighbors.model_name: OffenceKNeighbors,
    EfficiencyKNeighbors.model_name: EfficiencyKNeighbors,
}
