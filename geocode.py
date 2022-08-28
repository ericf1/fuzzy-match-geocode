import pandas as pd
import numpy as np
import time
from Levenshtein import jaro_winkler
import concurrent.futures
import pyarrow.parquet as pq
import pyarrow as pa
import gc
# must also pip install pyarrow
#  bsub -G compute-jskrastins -q general -n 500 -a 'docker(python:slim)' 'python /home/jskrastins/geocode_by_fuzzy_match.py'

# static values:
NUMBER_OF_CORES = 15
NUMBER_OF_CHUNKS = 20
read_list = [142, 132, 134, 136]
load_from = 1
load_to = 2000000

key_addresses_df = pd.read_parquet(
    'Brazil-Latest-points.parquet', engine='pyarrow')[["id", "housenumber", "street", "code_muni"]]
key_streets_df = pd.read_parquet(
    'Brazil-Latest-lines.parquet', engine='pyarrow')[["id", "street", "code_muni"]]
# merging the muni tables
muni_code_df = pd.read_parquet('muni_id.parquet', engine="pyarrow")[
    ["code_muni", "codigo_do_municipio_no_bcbase"]]
muni_code_df = muni_code_df.astype(
    {"codigo_do_municipio_no_bcbase": int}, errors="raise")


def point_housenumber_match(muni_df, addr):
    # matches directly the municipality and the housenumber using the points parquet
    addr = addr.iloc[0]
    # we first want to use the stnd number if avaliable
    stnd_numero = addr["stnd_numero"]
    if stnd_numero:
        return muni_df[(muni_df["housenumber"] == stnd_numero)]
    return muni_df[(muni_df["housenumber"] == addr["numero"])]


def point_muni_by_index(addr):
    return key_addresses_df[(key_addresses_df["code_muni"] == addr.iloc[0]["code_muni"])]


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
        cur_jaro_winkler_distance = jaro_winkler(
            (df_street_name), (street_name))
        if cur_jaro_winkler_distance >= distance_metric:
            code_muni = df["code_muni"].iloc[idx]
            id = df["id"].iloc[idx]
            return pd.DataFrame(data={"code_muni": code_muni, "id": id, "process": [3], "street": df["street"].iloc[idx], "jaro_distance": cur_jaro_winkler_distance})
    return None


def street_muni_by_index(addr):
    return key_streets_df[(key_streets_df["code_muni"] == addr.iloc[0]["code_muni"])]


def street_street_by_index(muni_df, addr):
    addr = addr.iloc[0]
    if not addr['stnd_type']:
        return
    street_name = f"{addr['stnd_type']} {addr['stnd_logr']}"
    return muni_df[(muni_df["street"] == street_name)]


def by_point(addr):
    muni_df = point_muni_by_index(addr)
    match = point_housenumber_match(muni_df, addr)
    if match is None or match.empty:
        return find_best_match(muni_df, addr, .95)
    match = match[:1]
    match["process"] = [1]
    return match


def by_street(addr):
    muni_df = street_muni_by_index(addr)
    match = street_street_by_index(muni_df, addr)
    if match is None or match.empty:
        return find_best_match(muni_df, addr, .95)
    match = match.iloc[:1]
    match["process"] = [2]
    return match


def perform_matches(addr):
    # points data
    points_street_match = by_point(addr)
    if not points_street_match is None:
        return [addr, points_street_match, "points_data"]
    # streets data
    street_only_match = by_street(addr)
    if not street_only_match is None:
        return [addr, street_only_match, "street_data"]
    # Both failed (Not similar enough to anything)
    return [addr, None]


def export_as_csv(df, name):
    return df.to_csv(name, index=False, header=True, encoding="utf-8")


def geocode_section(section_of_df, chunk_num, actual):
    start = time.perf_counter()
    df_output = pd.DataFrame()
    df_failed_addresses = pd.DataFrame()
    # number of threads allocated to this python script
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUMBER_OF_CORES) as executor:
        results = executor.map(perform_matches, [
            section_of_df.iloc[[i]] for i in range(0, len(section_of_df.index))])  # range(0, 100)
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

    export_as_csv(
        df_output, f'output/addresses_matched_{load_from+actual}_{load_to+actual}_{chunk_num}.csv')
    export_as_csv(df_failed_addresses,
                  f'output/failed_addresses_{load_from+actual}_{load_to+actual}_{chunk_num}.csv')
    print(f"This took {round(finish - start, 5)} seconds")


def main():
    for mil in read_list:
        actual = mil * 1000000
        need_to_geocode_df = pd.read_parquet(
            f'standardized_addr_officePC_{mil}.parquet', engine='pyarrow')[["stnd_type", "stnd_logr",
                                                                            "stnd_numero", "logradouro", "numero", "codigo_do_municipio_no_bcbase", "row_id"]]
        need_to_geocode_df = need_to_geocode_df.iloc[load_from:load_to, ]
        gc.collect()
        # ensuring type errors are not wrong
        need_to_geocode_df = need_to_geocode_df.astype(
            {"codigo_do_municipio_no_bcbase": int}, errors="raise")
        need_to_geocode_df = pd.merge(need_to_geocode_df, muni_code_df,
                                      on=["codigo_do_municipio_no_bcbase"])
        need_to_geocode_df_parts = np.array_split(
            need_to_geocode_df, NUMBER_OF_CHUNKS)
        for idx, df in enumerate(need_to_geocode_df_parts):
            print(
                f"Processing from row {load_from+actual} to {load_to+actual}. Chunk {idx} out of {NUMBER_OF_CHUNKS}.")
            geocode_section(df, idx, actual)


if __name__ == "__main__":
    main()
