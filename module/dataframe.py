import pandas as pd
import gc


def csv_to_parquet(csv_path, save_name):
    """해당코드는 Baseline에서 구현되어 있는 코드 입니다."""
    df = pd.read_csv(csv_path)
    df.to_parquet(f"./{save_name}")
    del df
    gc.collect()
    print(save_name, "Done.")


def df_query_by_type(df: pd.DataFrame, types=object):
    return df[df.columns[df.dtypes == types]]


def series_string_filter(series: pd.Series, strings: list[str], strict=False):
    __bool_df = series.apply(lambda _: False)
    if not strict:
        __bool_df = series.apply(lambda _X: sum([s in _X for s in strings]) > 0)
    elif strict:
        __bool_df = series.apply(lambda _X: sum([s in _X for s in strings]) == len(strings))

    return __bool_df


def pred_muti_model(x_test, models: list):
    results = pd.DataFrame()
    results[0] = models[0].predict(x_test)
    for index, model in enumerate(models[1:]):
        results[index + 1] = model.predict(x_test)
    return results
