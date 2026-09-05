"""Vectorized batch feature-matrix construction for whole slices.

Row-by-row `compute_features` is the safe reference but is O(n) history
scans per row — infeasible over hundreds of thousands of rows. This
module builds the same features in one forward pass. Its only claim to
correctness is that eval/test_leakage.py checks it against
`compute_features` on a sample; it must never be treated as
independently trustworthy.
"""
import pandas as pd

from src.features.behavioral import build_behavioral_matrix


def build_static_matrix(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["TransactionDT"]
    p_email = df.get("P_emaildomain")
    r_email = df.get("R_emaildomain")
    out = pd.DataFrame(index=df.index)
    out["amount"] = df["TransactionAmt"]
    out["hour_of_day"] = ((dt // 3600) % 24).astype(int)
    out["day_of_week"] = ((dt // 86400) % 7).astype(int)
    out["product_cd"] = df.get("ProductCD")
    for c in ["card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2", "dist1", "dist2"]:
        out[c] = df.get(c)
    out["has_dist1"] = df.get("dist1").notna().astype(int)
    out["has_dist2"] = df.get("dist2").notna().astype(int)
    out["p_email_domain"] = p_email
    out["r_email_domain"] = r_email
    out["email_domain_match"] = (p_email.notna() & (p_email == r_email)).astype(int)
    out["device_type"] = df.get("DeviceType")
    out["device_info"] = df.get("DeviceInfo")
    out["os"] = df.get("id_30")
    out["browser"] = df.get("id_31")
    out["screen_res"] = df.get("id_33")
    return out


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """df must already be sorted by TransactionDT and carry
    card_entity/device_entity (src/data/entities.add_entity_keys)."""
    static = build_static_matrix(df)
    behavioral = build_behavioral_matrix(df)
    matrix = pd.concat([static, behavioral], axis=1)
    matrix.insert(0, "TransactionID", df["TransactionID"].values)
    matrix.insert(1, "isFraud", df["isFraud"].values)
    matrix.insert(2, "TransactionDT", df["TransactionDT"].values)
    return matrix
