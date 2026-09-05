"""Phase 8 input contract. The analyst layer receives extracted
evidence only — never the raw transaction, never the thresholds."""
from dataclasses import dataclass, field


@dataclass
class FeatureContribution:
    name: str
    value: float
    contribution: float


@dataclass
class AnalystEvidence:
    risk_score: float
    top_features: list  # list[FeatureContribution]
    decision: str
    transaction_id: str = None

    def missing_fields(self) -> list:
        missing = []
        if self.risk_score is None:
            missing.append("risk_score")
        if not self.decision:
            missing.append("decision")
        if not self.top_features:
            missing.append("top_features")
        else:
            for i, feat in enumerate(self.top_features):
                name = feat.get("name") if isinstance(feat, dict) else feat.name
                value = feat.get("value") if isinstance(feat, dict) else feat.value
                contribution = feat.get("contribution") if isinstance(feat, dict) else feat.contribution
                if name is None:
                    missing.append(f"top_features[{i}].name")
                if value is None:
                    missing.append(f"top_features[{i}].value")
                if contribution is None:
                    missing.append(f"top_features[{i}].contribution")
        return missing
