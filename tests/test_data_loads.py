import pandas as pd
from src.data_loads import load_data


def test_load_data_returns_dataframe():
    df = load_data()
    assert isinstance(df, pd.DataFrame)


def test_load_data_not_empty():
    df = load_data()
    assert not df.empty


def test_load_data_has_expected_columns():
    df = load_data()

    expected_columns = {
        "age",
        "gender",
        "job_type",
        "daily_social_media_time",
        "social_platform_preference",
        "number_of_notifications",
        "work_hours_per_day",
        "perceived_productivity_score",
        "actual_productivity_score",
        "stress_level",
        "sleep_hours",
        "screen_time_before_sleep",
        "breaks_during_work",
        "uses_focus_apps",
        "has_digital_wellbeing_enabled",
        "coffee_consumption_per_day",
        "days_feeling_burnout_per_month",
        "weekly_offline_hours",
        "job_satisfaction_score",
    }

    assert expected_columns.issubset(set(df.columns))