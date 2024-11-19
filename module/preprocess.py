from module.dataframe import df_query_by_type, series_string_filter

import pandas as pd
import numpy as np
import holidays
from sklearn.preprocessing import LabelEncoder

kr_holidays = holidays.KR()

tourism_road = ["관광단지", "관광도로"]
general_road = ["지방도", "일반국도", "길", "로"]
bridge_road = ["교", "천교"]
nontaged_road = ["-"]


def mapping_road(road_name):
    if sum([target in road_name for target in tourism_road]) > 0:
        return "tourism_road"
    elif sum([target in road_name for target in general_road]) > 0:
        return "general_road"
    elif sum([target in road_name for target in bridge_road]) > 0:
        return "bridge"
    elif sum([target in road_name for target in nontaged_road]) > 0:
        return "-"
    else:
        return "-"


def to_date_str(int_date: int):
    day = int_date % 100
    year = (int_date - day) // 10000
    month = (int_date - year * 10000) // 100
    return f"{year}-{month:02}-{day:02}"


def do_preprocessing(train, test):
    # non-unique 제거하기
    is_unique = train.apply(pd.Series.unique, axis=0).apply(len)
    is_unique[is_unique == len(train)].index, is_unique[is_unique == 1].index
    train = train.drop(columns=is_unique[is_unique == 1].index)
    test = test.drop(columns=is_unique[is_unique == 1].index)

    # weight_restricted - label추가하기
    train["weight_restricted"] = (train["weight_restricted"] != 0).astype(int)
    test["weight_restricted"] = (test["weight_restricted"] != 0).astype(int)

    # is_holiday - label추가하기
    train["is_holiday"] = train["base_date"].apply(to_date_str)
    test["is_holiday"] = test["base_date"].apply(to_date_str)
    train["is_holiday"] = train["is_holiday"].apply(lambda _X: _X in kr_holidays).astype(int)
    test["is_holiday"] = test["is_holiday"].apply(lambda _X: _X in kr_holidays).astype(int)

    # road_type - label추가하기
    train["road_type"] = train["road_name"].apply(mapping_road)
    test["road_type"] = test["road_name"].apply(mapping_road)

    # label encode.
    str_col = [
        "day_of_week",
        "start_turn_restricted",
        "end_turn_restricted",
        "road_type",
        "is_holiday",
        "weight_restricted",
    ]
    for col in str_col:
        label_enc = LabelEncoder()
        label_enc = label_enc.fit(train[col])
        train[col] = label_enc.transform(train[col])

        for label in np.unique(test[col]):
            if label not in label_enc.classes_:
                label_enc.classes_ = np.append(label_enc.classes_, label)
        test[col] = label_enc.transform(test[col])

    # One-Hot Enc
    train = pd.get_dummies(
        train,
        columns=str_col,
        dtype=int,
    )
    test = pd.get_dummies(
        test,
        columns=str_col,
        dtype=int,
    )

    # 날짜 삽입입
    train["base_date_year"] = train["base_date"] // 10000
    train["base_date_month"] = train["base_date"] // 100 % 100
    train["base_date_day"] = train["base_date"] % 100
    train = train.drop(columns="base_date")

    test["base_date_year"] = test["base_date"] // 10000
    test["base_date_month"] = test["base_date"] // 100 % 100
    test["base_date_day"] = test["base_date"] % 100
    test = test.drop(columns="base_date")

    # 문자형 데이터 제거하기.
    train = pd.concat([df_query_by_type(train, float), df_query_by_type(train, int)], axis=1)
    test = pd.concat([df_query_by_type(test, float), df_query_by_type(test, int)], axis=1)

    # 좌표 정보 제거하기.
    train = train.drop(columns=["start_latitude", "start_longitude", "end_latitude", "end_longitude"])
    test = test.drop(columns=["start_latitude", "start_longitude", "end_latitude", "end_longitude"])

    return train, test
