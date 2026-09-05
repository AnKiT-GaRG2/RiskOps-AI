"""Static features — no history required, no as_of risk."""
import pandas as pd

STATIC_RAW_COLS = [
    "TransactionAmt", "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "dist1", "dist2",
    "P_emaildomain", "R_emaildomain",
    "DeviceType", "DeviceInfo", "id_30", "id_31", "id_33",
]


def compute_static_features(txn: pd.Series) -> dict:
    dt = txn["TransactionDT"]
    return {
        "amount": txn.get("TransactionAmt"),
        "hour_of_day": int((dt // 3600) % 24),
        "day_of_week": int((dt // 86400) % 7),
        "product_cd": txn.get("ProductCD"),
        "card1": txn.get("card1"),
        "card2": txn.get("card2"),
        "card3": txn.get("card3"),
        "card4": txn.get("card4"),
        "card5": txn.get("card5"),
        "card6": txn.get("card6"),
        "addr1": txn.get("addr1"),
        "addr2": txn.get("addr2"),
        "dist1": txn.get("dist1"),
        "dist2": txn.get("dist2"),
        "has_dist1": int(pd.notna(txn.get("dist1"))),
        "has_dist2": int(pd.notna(txn.get("dist2"))),
        "p_email_domain": txn.get("P_emaildomain"),
        "r_email_domain": txn.get("R_emaildomain"),
        "email_domain_match": int(
            pd.notna(txn.get("P_emaildomain"))
            and txn.get("P_emaildomain") == txn.get("R_emaildomain")
        ),
        "device_type": txn.get("DeviceType"),
        "device_info": txn.get("DeviceInfo"),
        "os": txn.get("id_30"),
        "browser": txn.get("id_31"),
        "screen_res": txn.get("id_33"),
    }
