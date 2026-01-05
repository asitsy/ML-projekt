from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "social_media_vs_productivity.csv"

CATEGORICAL_FEATURES = [
    "gender",
    "job_type",
    "social_platform_preference",
    "uses_focus_apps",
    "has_digital_wellbeing_enabled",
]

NUMERIC_FEATURES = [
    "age",
    "daily_social_media_time",
    "number_of_notifications",
    "work_hours_per_day",
    "perceived_productivity_score",
    "stress_level",
    "sleep_hours",
    "screen_time_before_sleep",
    "breaks_during_work",
    "coffee_consumption_per_day",
    "days_feeling_burnout_per_month",
    "weekly_offline_hours",
    "job_satisfaction_score",
]
