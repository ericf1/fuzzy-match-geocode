import pandas as pd
import numpy as np
import time
from Levenshtein import jaro_winkler
import concurrent.futures
# must also pip install pyarrow

need_to_geocode_df = pd.read_parquet(
    'standardized_addr_eric.parquet', engine='pyarrow')[["stnd_type", "stnd_logr",
                                                        "stnd_numero", "logradouro", "numero", "codigo_do_municipio_no_bcbase", "row_id"]]
key_addresses_df = pd.read_parquet(
    'Brazil-Latest-points.parquet', engine='pyarrow')[["id", "housenumber", "street", "code_muni"]]
key_streets_df = pd.read_parquet(
    'Brazil-Latest-lines.parquet', engine='pyarrow')[["id", "street", "code_muni"]]
# merging the muni tables
muni_code_df = pd.read_parquet('muni_id.parquet', engine="pyarrow")[
    ["code_muni", "codigo_do_municipio_no_bcbase"]]
# ensuring type errors are not wrong
need_to_geocode_df = need_to_geocode_df.astype(
    {"codigo_do_municipio_no_bcbase": int}, errors="raise")
muni_code_df = muni_code_df.astype(
    {"codigo_do_municipio_no_bcbase": int}, errors="raise")
need_to_geocode_df = pd.merge(need_to_geocode_df, muni_code_df,
                              on=["codigo_do_municipio_no_bcbase"])

# static values:
NUMBER_OF_CORES = 1
NUMBER_OF_CHUNKS = 10


def perfect_match_by_muni_housenumber(addr):
    # matches directly the municipality and the housenumber using the points parquet
    addr = addr.iloc[0]
    # we first want to use the stnd number if avaliable
    stnd_numero = addr["stnd_numero"]
    if stnd_numero:
        return key_addresses_df[(key_addresses_df["housenumber"] == stnd_numero) & (
            key_addresses_df["code_muni"] == addr["code_muni"])]
    return key_addresses_df[(key_addresses_df["housenumber"] == addr["numero"]) & (
        key_addresses_df["code_muni"] == addr["code_muni"])]


def perfect_match_by_muni(addr):
    # matches directly only by the municipality using the lines parquet
    return key_streets_df[(key_streets_df["code_muni"] == addr.iloc[0]["code_muni"])]


def stnd_parse_str(street_name):
    # parsing the street name string for redundant info
    return str(street_name).replace("Rua ", "").replace("Avenida ", "")


def find_best_match(df, addr, distance_metric):
    # given a df and an address, finds the closest matching street name with at least the given distance value
    addr = addr.iloc[0]
    if addr["stnd_logr"]:
        street_name = addr["stnd_logr"]
    else:
        street_name = addr["logradouro"]

    for idx in range(0, df.shape[0], 1):
        df_street_name = stnd_parse_str(df["street"].iloc[idx])
        # if df_street_name[0] != street_name[0]:
        # continue
        # if not (len(df_street_name) - 5 <= len(street_name) <= len(df_street_name) + 5):
        # continue
        cur_jaro_winkler_distance = jaro_winkler(
            (df_street_name), (street_name))
        if cur_jaro_winkler_distance >= distance_metric:
            code_muni = df["code_muni"].iloc[idx]
            id = df["id"].iloc[idx]
            return pd.DataFrame(data={"code_muni": code_muni, "id": id, "process": [3], "street": df["street"].iloc[idx], "jaro_distance": cur_jaro_winkler_distance})
            # return pd.DataFrame(data={"code_muni": code_muni, "id": id, "process": [3]})
    return None


def perfect_match_by_street_name(addr):
    addr = addr.iloc[0]
    if not addr['stnd_type']:
        return
    street_name = f"{addr['stnd_type']} {addr['stnd_logr']}"
    match = key_streets_df[(key_streets_df["street"] == street_name) & (
        key_streets_df["code_muni"] == addr["code_muni"])]
    if match.empty:
        return None
    return match


def perfect_match_by_street_name_points_data(addr, df):
    addr = addr.iloc[0]
    if not addr['stnd_type']:
        return
    street_name = f"{addr['stnd_type']} {addr['stnd_logr']}"
    match = df[(df["street"] == street_name)]
    if match.empty:
        return
    match = match.iloc[:1]
    match["process"] = [1]
    return match


def points_data_match(addr):
    # fuzzy matching the address
    df = perfect_match_by_muni_housenumber(addr)
    if df.empty:
        return None
    perfect_match = perfect_match_by_street_name_points_data(addr, df)
    if perfect_match is not None:
        return perfect_match
    return find_best_match(df, addr, .95)


def streets_data_match(addr):
    # fuzzy matching only the street and using the lines parquet
    df = perfect_match_by_muni(addr)
    # if we can't match the muni then, we should just return
    if df.empty:
        return None
    output = perfect_match_by_street_name(addr)
    if not output is None:
        # restricts to only one street match
        output = output.iloc[:1]
        output["process"] = [2]
        return output
    return find_best_match(df, addr, .95)


def perform_matches(addr):
    # points data
    points_street_match = points_data_match(addr)
    if not points_street_match is None:
        return [addr, points_street_match, "points_data"]
    # streets data
    street_only_match = streets_data_match(addr)
    if not street_only_match is None:
        return [addr, street_only_match, "street_data"]
    # Both failed (Not similar enough to anything)
    return [addr, None]


def export_as_csv(df, name):
    return df.to_csv(name, index=False, header=True, encoding="utf-8")


def geocode_section(section_of_df, chunk_num):
    start = time.perf_counter()
    df_output = pd.DataFrame()
    df_failed_addresses = pd.DataFrame()
    # number of threads allocated to this python script
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUMBER_OF_CORES) as executor:
        results = executor.map(perform_matches, [
            section_of_df.iloc[[i]] for i in range(0, 1)])
        for result in results:
            df_sent = result[0]
            df_most_similar = result[1]
            if df_most_similar is None or df_most_similar.empty:
                # there were not matching streets that were close enough
                df_sent["stnd_addr_index"] = df_sent.iloc[0]["row_id"]
                df_failed_addresses = pd.concat([df_failed_addresses, df_sent])
                continue
            df_most_similar["standardized_addr_index"] = df_sent.iloc[0]["row_id"]
            df_most_similar["dataset"] = result[2]
            original_street_name = df_sent.iloc[0]["stnd_logr"]
            if not original_street_name:
                original_street_name = df_sent.iloc[0]["logradouro"]
            df_most_similar["original_street_name"] = original_street_name
            df_output = pd.concat([df_output, df_most_similar])
    finish = time.perf_counter()
    df_output = df_output.reset_index(drop=True)
    print(df_output)

    export_as_csv(df_output, f'addresses_matched{chunk_num}.csv')
    export_as_csv(df_failed_addresses, f'failed_addresses{chunk_num}.csv')
    print(f"This took {round(finish - start, 5)} seconds")


def main():
    need_to_geocode_df_parts = np.array_split(
        need_to_geocode_df, NUMBER_OF_CHUNKS)
    for idx, df in enumerate(need_to_geocode_df_parts):
        geocode_section(df, idx)


if __name__ == "__main__":
    main()
