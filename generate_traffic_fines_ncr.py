"""
Generate a realistic synthetic dataset (traffic_fines_ncr.csv) for predicting
traffic-fine risk across 40 Delhi NCR checkpoints (Jan 1, 2025 - Jun 30, 2025).

Dependencies: pandas, numpy only (seed=42).
"""

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)


def build_checkpoints() -> pd.DataFrame:
    """Create 40 real Delhi NCR junctions/roads with approximate coordinates and attributes."""
    # Columns: id, name, lat, lon, road_type, speed_limit, near_metro, police, camera, traffic
    data = [
        ("CP01", "Connaught Place (Rajiv Chowk)", 28.6315, 77.2167, "signal", 50, 1, 0.90, 0.92, 0.95),
        ("CP02", "ITO Junction", 28.6289, 77.2414, "signal", 50, 1, 0.92, 0.90, 0.94),
        ("CP03", "Dhaula Kuan", 28.5921, 77.1617, "flyover", 70, 1, 0.85, 0.88, 0.92),
        ("CP04", "AIIMS Flyover", 28.5672, 77.2100, "flyover", 60, 1, 0.82, 0.85, 0.90),
        ("CP05", "Anand Vihar ISBT", 28.6469, 77.3160, "signal", 50, 1, 0.86, 0.78, 0.91),
        ("CP06", "Akshardham Crossing", 28.6127, 77.2773, "highway", 80, 1, 0.78, 0.86, 0.85),
        ("CP07", "IFFCO Chowk Gurugram", 28.4776, 77.0725, "highway", 80, 1, 0.84, 0.88, 0.91),
        ("CP08", "Noida Sector 18", 28.5708, 77.3261, "market", 40, 1, 0.80, 0.82, 0.89),
        ("CP09", "Kashmere Gate ISBT", 28.6675, 77.2282, "ring_road", 60, 1, 0.88, 0.84, 0.93),
        ("CP10", "Nehru Place", 28.5494, 77.2519, "market", 40, 1, 0.76, 0.79, 0.86),
        ("CP11", "Mahipalpur Chowk", 28.5453, 77.1292, "highway", 70, 0, 0.81, 0.83, 0.88),
        ("CP12", "Dwarka Mor", 28.6193, 77.0333, "signal", 50, 1, 0.70, 0.65, 0.80),
        ("CP13", "Pari Chowk Greater Noida", 28.4652, 77.5105, "signal", 60, 1, 0.68, 0.70, 0.74),
        ("CP14", "Rajouri Garden", 28.6492, 77.1215, "ring_road", 60, 1, 0.79, 0.76, 0.87),
        ("CP15", "Lajpat Nagar", 28.5677, 77.2433, "market", 40, 1, 0.77, 0.74, 0.88),
        ("CP16", "Karol Bagh (Pusa Road)", 28.6514, 77.1907, "market", 40, 1, 0.75, 0.72, 0.85),
        ("CP17", "Ashram Chowk", 28.5714, 77.2591, "flyover", 60, 1, 0.87, 0.85, 0.95),
        ("CP18", "Moolchand Flyover", 28.5643, 77.2342, "flyover", 60, 1, 0.80, 0.82, 0.88),
        ("CP19", "Sarai Kale Khan", 28.5892, 77.2597, "ring_road", 70, 1, 0.83, 0.81, 0.89),
        ("CP20", "Peeragarhi Chowk", 28.6798, 77.0928, "ring_road", 60, 1, 0.76, 0.71, 0.86),
        ("CP21", "Mukarba Chowk", 28.7365, 77.1528, "highway", 80, 0, 0.78, 0.75, 0.84),
        ("CP22", "Azadpur Mandi", 28.7076, 77.1755, "ring_road", 60, 1, 0.74, 0.68, 0.89),
        ("CP23", "Punjabi Bagh Chowk", 28.6689, 77.1282, "ring_road", 60, 1, 0.79, 0.77, 0.87),
        ("CP24", "Janakpuri District Centre", 28.6289, 77.0815, "signal", 50, 1, 0.72, 0.70, 0.82),
        ("CP25", "Subhash Nagar More", 28.6406, 77.1050, "signal", 50, 1, 0.69, 0.66, 0.79),
        ("CP26", "Hauz Khas (Outer Ring Rd)", 28.5494, 77.2001, "ring_road", 60, 1, 0.81, 0.83, 0.86),
        ("CP27", "Saket (Press Enclave)", 28.5245, 77.2066, "signal", 50, 1, 0.74, 0.76, 0.81),
        ("CP28", "Badarpur Border", 28.4933, 77.3033, "highway", 70, 1, 0.85, 0.79, 0.88),
        ("CP29", "Cyber Hub Gurugram", 28.4950, 77.0895, "highway", 70, 1, 0.83, 0.89, 0.87),
        ("CP30", "Rajiv Chowk Gurugram", 28.4495, 77.0328, "highway", 80, 0, 0.80, 0.84, 0.86),
        ("CP31", "Huda City Centre Gurugram", 28.4593, 77.0724, "signal", 50, 1, 0.77, 0.80, 0.83),
        ("CP32", "Sohna Road Subhash Chowk", 28.4285, 77.0371, "flyover", 60, 0, 0.71, 0.73, 0.79),
        ("CP33", "Botanical Garden Noida", 28.5641, 77.3342, "signal", 50, 1, 0.78, 0.77, 0.85),
        ("CP34", "Noida City Centre (Sec 32)", 28.5747, 77.3560, "signal", 50, 1, 0.73, 0.74, 0.80),
        ("CP35", "DND Flyway Toll Plaza", 28.5724, 77.2848, "flyover", 80, 0, 0.82, 0.91, 0.89),
        ("CP36", "Mahamaya Flyover Noida", 28.5469, 77.3210, "flyover", 70, 0, 0.75, 0.80, 0.84),
        ("CP37", "Vaishali Metro Crossing", 28.6499, 77.3398, "signal", 50, 1, 0.72, 0.67, 0.81),
        ("CP38", "Indirapuram (NH-24)", 28.6366, 77.3694, "highway", 80, 0, 0.74, 0.76, 0.83),
        ("CP39", "Kaushambi Border", 28.6454, 77.3243, "signal", 50, 1, 0.79, 0.72, 0.86),
        ("CP40", "Bata Chowk Faridabad", 28.3852, 77.3133, "highway", 70, 1, 0.70, 0.68, 0.78),
    ]
    cols = [
        "checkpoint_id", "checkpoint_name", "latitude", "longitude",
        "road_type", "speed_limit_kmph", "is_near_metro",
        "police_presence", "camera_density", "traffic_density",
    ]
    return pd.DataFrame(data, columns=cols)


def simulate_weather_and_temp(months: np.ndarray, hours: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generate realistic Delhi NCR weather and temperature_c for Jan-Jun."""
    n = len(months)
    weather = np.empty(n, dtype=object)

    # Monthly weather probabilities: [clear, rain, fog, smog]
    probs_by_month = {
        1: [0.35, 0.05, 0.35, 0.25],  # Jan: fog & winter smog
        2: [0.60, 0.06, 0.14, 0.20],  # Feb: milder, lingering fog/smog
        3: [0.80, 0.05, 0.02, 0.13],  # Mar: mostly clear
        4: [0.85, 0.04, 0.00, 0.11],  # Apr: warm & clear
        5: [0.82, 0.07, 0.00, 0.11],  # May: hot, occasional dust/rain
        6: [0.68, 0.24, 0.00, 0.08],  # Jun: pre-monsoon / monsoon rain
    }
    labels = np.array(["clear", "rain", "fog", "smog"])
    for m, p in probs_by_month.items():
        mask = months == m
        # Fog is much more likely at night / early morning (22-09)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        p_arr = np.tile(p, (len(idx), 1))
        night_early = (hours[idx] <= 9) | (hours[idx] >= 22)
        if p[2] > 0:
            p_arr[night_early, 2] *= 1.6
            p_arr[~night_early, 2] *= 0.25
            p_arr = p_arr / p_arr.sum(axis=1, keepdims=True)
        # Vectorized categorical draw via cumulative probabilities
        u = rng.random(len(idx))[:, None]
        cum_p = np.cumsum(p_arr, axis=1)
        choice_idx = (u > cum_p).sum(axis=1)
        weather[idx] = labels[choice_idx]

    # Monthly base mean & diurnal amplitude in Celsius
    mean_temp_map = {1: 13.5, 2: 17.5, 3: 23.5, 4: 30.5, 5: 35.0, 6: 35.5}
    amp_map = {1: 5.5, 2: 6.0, 3: 7.0, 4: 7.5, 5: 7.0, 6: 5.5}
    base_temp = np.array([mean_temp_map[m] for m in months])
    amp = np.array([amp_map[m] for m in months])

    # Diurnal cycle: coolest ~05:00, warmest ~15:00
    diurnal = -amp * np.cos(2 * np.pi * (hours - 5) / 24.0)
    rain_cool = np.where(weather == "rain", -3.2, 0.0)
    fog_cool = np.where(weather == "fog", -1.8, 0.0)
    noise = rng.normal(0.0, 1.8, size=n)
    temp_c = np.round(base_temp + diurnal + rain_cool + fog_cool + noise, 1)
    return weather, temp_c


def main() -> None:
    checkpoints = build_checkpoints()

    # Full hourly grid from 2025-01-01 00:00 to 2025-06-30 23:00 (181 days * 24h = 4,344 hours)
    all_dates = pd.date_range("2025-01-01", "2025-06-30", freq="D")
    n_days = len(all_dates)
    hours_per_day = 24
    total_hours = n_days * hours_per_day  # 4,344

    # Sample 1,500 unique hours per checkpoint -> 40 * 1,500 = 60,000 rows
    samples_per_cp = 1500
    n_cp = len(checkpoints)

    cp_indices = np.repeat(np.arange(n_cp), samples_per_cp)
    hour_slot_indices = np.concatenate([
        np.sort(rng.choice(total_hours, size=samples_per_cp, replace=False))
        for _ in range(n_cp)
    ])

    day_idx = hour_slot_indices // hours_per_day
    hour = (hour_slot_indices % hours_per_day).astype(int)
    date_dt = all_dates[day_idx]
    date_str = date_dt.strftime("%Y-%m-%d")
    day_of_week = date_dt.dayofweek.to_numpy()
    is_weekend = (day_of_week >= 5).astype(int)
    month = date_dt.month.to_numpy()

    # Major Delhi NCR public holidays / festival days in Jan-Jun 2025
    holiday_dates = {
        "2025-01-01",  # New Year's Day
        "2025-01-14",  # Makar Sankranti / Pongal
        "2025-01-26",  # Republic Day
        "2025-02-26",  # Maha Shivaratri
        "2025-03-14",  # Holi
        "2025-03-31",  # Eid-ul-Fitr
        "2025-04-06",  # Ram Navami
        "2025-04-10",  # Mahavir Jayanti
        "2025-04-14",  # Ambedkar Jayanti / Baisakhi
        "2025-04-18",  # Good Friday
        "2025-05-12",  # Buddha Purnima
        "2025-06-07",  # Eid-ul-Adha
    }
    is_holiday = np.isin(date_str, list(holiday_dates)).astype(int)

    df = checkpoints.iloc[cp_indices].reset_index(drop=True)
    df["date"] = date_str
    df["hour"] = hour
    df["day_of_week"] = day_of_week
    df["is_weekend"] = is_weekend
    df["is_holiday"] = is_holiday
    df["month"] = month

    n = len(df)
    weather, temp_c = simulate_weather_and_temp(month, hour)
    df["weather"] = weather
    df["temperature_c"] = temp_c

    # Nearby events (more frequent near markets, metro hubs, weekends, holidays)
    event_prob = (
        0.04
        + 0.05 * df["is_near_metro"].to_numpy()
        + 0.06 * (df["road_type"].to_numpy() == "market")
        + 0.05 * is_weekend
        + 0.08 * is_holiday
    )
    event_prob = np.clip(event_prob, 0.02, 0.35)
    df["event_nearby"] = (rng.random(n) < event_prob).astype(int)

    # Checkpoint active probability: high during day/peak, slightly lower late night & rain/fog
    is_late_night = np.isin(hour, [2, 3, 4, 5])
    is_drunk_window = np.isin(hour, [22, 23, 0, 1, 2])
    active_prob = (
        0.78
        + 0.18 * df["police_presence"].to_numpy()
        - np.where(is_late_night, 0.30, 0.0)
        + np.where(is_drunk_window & np.isin(day_of_week, [4, 5]), 0.08, 0.0)
        - np.where(weather == "rain", 0.12, 0.0)
        - np.where(weather == "fog", 0.10, 0.0)
        + 0.05 * is_holiday
    )
    active_prob = np.clip(active_prob, 0.15, 0.99)
    checkpoint_active = (rng.random(n) < active_prob).astype(int)
    df["checkpoint_active"] = checkpoint_active

    # Time-of-day traffic & enforcement multiplier
    is_morning_peak = (hour >= 8) & (hour <= 11)
    is_evening_peak = (hour >= 17) & (hour <= 21)
    is_office_hour = (hour >= 8) & (hour <= 19)

    time_mult = np.ones(n)
    time_mult = np.where(is_morning_peak, 1.45, time_mult)
    time_mult = np.where(is_evening_peak, 1.55, time_mult)
    time_mult = np.where(is_drunk_window, 0.85, time_mult)
    time_mult = np.where(np.isin(hour, [3, 4, 5]), 0.35, time_mult)

    # Weekends reduce office-hour traffic/fines, boost evening/night leisure checks
    weekend_mult = np.where(
        (is_weekend == 1) & is_office_hour,
        0.75,
        np.where((is_weekend == 1) & is_drunk_window, 1.25, 1.0),
    )

    # Special drives: Jan drunk-driving / New Year drive + holiday/event surges
    special_drive = np.where((month == 1) & is_drunk_window, 1.35, 1.0)
    holiday_mult = np.where(is_holiday == 1, 1.25, 1.0)
    event_mult = np.where(df["event_nearby"].to_numpy() == 1, 1.18, 1.0)
    weather_mult = np.where(
        weather == "rain", 0.86, np.where(weather == "fog", 0.88, 1.0)
    )

    police = df["police_presence"].to_numpy()
    camera = df["camera_density"].to_numpy()
    traffic = df["traffic_density"].to_numpy()

    # Vehicles checked (0 if checkpoint inactive)
    checked_lambda = (
        160.0
        * (0.35 + 0.65 * police)
        * (0.30 + 0.70 * traffic)
        * time_mult
        * weekend_mult
        * weather_mult
        * holiday_mult
        * rng.normal(1.0, 0.08, size=n).clip(0.6, 1.5)
    )
    vehicles_checked = np.where(
        checkpoint_active == 1,
        rng.poisson(np.maximum(checked_lambda, 1.0)),
        0,
    )
    df["vehicles_checked"] = vehicles_checked

    # Expected fines_issued rises with police_presence, camera_density, traffic_density
    fines_lambda = (
        18.0
        * (0.25 + 0.90 * police)
        * (0.25 + 0.85 * camera)
        * (0.25 + 0.90 * traffic)
        * time_mult
        * weekend_mult
        * special_drive
        * holiday_mult
        * event_mult
        * weather_mult
        * rng.normal(1.0, 0.12, size=n).clip(0.5, 1.6)
    )
    fines_issued = np.where(
        checkpoint_active == 1,
        rng.poisson(np.maximum(fines_lambda, 0.5)),
        0,
    )
    df["fines_issued"] = fines_issued

    # Violation category weights (7 categories summing to fines_issued)
    # Order: helmet, no_licence, drunk_driving, speeding, signal_jumping, wrong_parking, no_documents
    road_type = df["road_type"].to_numpy()
    w_helmet = np.full(n, 2.2) + 0.5 * (road_type == "signal")
    w_no_lic = np.full(n, 1.1)
    w_drunk = np.full(n, 0.15)
    w_speed = np.full(n, 1.2) + 1.8 * np.isin(road_type, ["highway", "flyover"]) + 0.6 * camera
    w_signal = np.full(n, 1.4) + 1.6 * (road_type == "signal") + 0.5 * camera
    w_parking = np.full(n, 1.0) + 2.2 * (road_type == "market") + 0.6 * df["is_near_metro"].to_numpy()
    w_docs = np.full(n, 1.3)

    # Boost drunk driving checks at 22-02, especially Fri/Sat and Jan special drives
    fri_sat = np.isin(day_of_week, [4, 5])
    drunk_boost = np.where(is_drunk_window, 2.8, 0.0)
    drunk_boost = np.where(is_drunk_window & fri_sat, drunk_boost * 1.9, drunk_boost)
    drunk_boost = np.where(is_drunk_window & (month == 1), drunk_boost * 1.5, drunk_boost)
    w_drunk = w_drunk + drunk_boost

    weights = np.column_stack([
        w_helmet, w_no_lic, w_drunk, w_speed, w_signal, w_parking, w_docs
    ])
    # Add slight per-row Dirichlet-like noise to weights
    weights = weights * rng.uniform(0.85, 1.15, size=weights.shape)
    probs = weights / weights.sum(axis=1, keepdims=True)

    # Multinomial draw for each row so the violation counts sum exactly to fines_issued
    viol_counts = np.zeros((n, 7), dtype=int)
    active_rows = np.where(fines_issued > 0)[0]
    for idx in active_rows:
        viol_counts[idx] = rng.multinomial(fines_issued[idx], probs[idx])

    viol_cols = [
        "helmet_violations",
        "no_licence_violations",
        "drunk_driving_violations",
        "speeding_violations",
        "signal_jumping_violations",
        "wrong_parking_violations",
        "no_documents_violations",
    ]
    viol_names = np.array([
        "helmet",
        "no_licence",
        "drunk_driving",
        "speeding",
        "signal_jumping",
        "wrong_parking",
        "no_documents",
    ])
    for j, col in enumerate(viol_cols):
        df[col] = viol_counts[:, j]

    # Top violation: largest count among the violation columns ("none" when fines_issued == 0)
    top_idx = np.argmax(viol_counts, axis=1)
    df["top_violation"] = np.where(fines_issued > 0, viol_names[top_idx], "none")

    # Realistic Indian challan amounts (INR):
    # helmet: 1000, no_licence: 5000, drunk_driving: 10000,
    # speeding: 1000-2000, signal_jumping: 5000, wrong_parking: 500-1000, no_documents: 500
    speeding_rate = rng.choice([1000, 1500, 2000], size=n)
    parking_rate = rng.choice([500, 1000], size=n)
    total_fine_inr = (
        viol_counts[:, 0] * 1000
        + viol_counts[:, 1] * 5000
        + viol_counts[:, 2] * 10000
        + viol_counts[:, 3] * speeding_rate
        + viol_counts[:, 4] * 5000
        + viol_counts[:, 5] * parking_rate
        + viol_counts[:, 6] * 500
    )
    df["total_fine_amount_inr"] = total_fine_inr.astype(int)

    # fine_risk_score: min-max scaled fines_issued per checkpoint (0-100)
    cp_min = df.groupby("checkpoint_id")["fines_issued"].transform("min")
    cp_max = df.groupby("checkpoint_id")["fines_issued"].transform("max")
    denom = (cp_max - cp_min).replace(0, 1)
    df["fine_risk_score"] = np.round(100.0 * (df["fines_issued"] - cp_min) / denom, 2)

    # ~1% missing values in weather and temperature_c only
    weather_missing_mask = rng.random(n) < 0.01
    temp_missing_mask = rng.random(n) < 0.01
    df.loc[weather_missing_mask, "weather"] = np.nan
    df.loc[temp_missing_mask, "temperature_c"] = np.nan

    # Enforce exact column order
    exact_columns = [
        "checkpoint_id",
        "checkpoint_name",
        "latitude",
        "longitude",
        "road_type",
        "speed_limit_kmph",
        "is_near_metro",
        "police_presence",
        "camera_density",
        "traffic_density",
        "date",
        "hour",
        "day_of_week",
        "is_weekend",
        "is_holiday",
        "month",
        "weather",
        "temperature_c",
        "event_nearby",
        "checkpoint_active",
        "vehicles_checked",
        "fines_issued",
        "total_fine_amount_inr",
        "top_violation",
        "helmet_violations",
        "no_licence_violations",
        "drunk_driving_violations",
        "speeding_violations",
        "signal_jumping_violations",
        "wrong_parking_violations",
        "no_documents_violations",
        "fine_risk_score",
    ]
    df = df[exact_columns]

    print("df.head():")
    print(df.head())
    print("\ndf.shape:", df.shape)
    print("\ndf.describe():")
    print(df.describe())

    df.to_csv("traffic_fines_ncr.csv", index=False)
    print("\nSaved traffic_fines_ncr.csv successfully.")


if __name__ == "__main__":
    main()