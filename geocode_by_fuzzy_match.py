import pandas as pd
import time
from Levenshtein import jaro_winkler
import concurrent.futures
# must also pip install pyarrow

need_to_geocode_df = pd.read_parquet(
    'standardized_addr_eric.parquet', engine='pyarrow')
key_addresses_df = pd.read_parquet(
    'Brazil-Latest-points.parquet', engine='pyarrow')
key_streets_df = pd.read_parquet(
    'Brazil-Latest-lines.parquet', engine='pyarrow')


def perfect_match_by_muni_housenumber(addr):
    # matches directly the municipality and the housenumber using the points parquet
    addr = addr.iloc[0]
    if addr["stnd_numero"]:
        housenumber = addr["stnd_numero"]
    else:
        housenumber = addr["numero"]
    match = key_addresses_df[(key_addresses_df["housenumber"] == housenumber) & (
        key_addresses_df["name_muni"] == addr["muni"]) & (key_addresses_df["name_state"] == addr["state"])]
    return match


def perfect_match_by_muni(addr):
    # matches directly only by the municipality using the lines parquet
    return key_streets_df[(key_streets_df["name_muni"] == addr.iloc[0]["muni"])]


def stnd_parse_str(street_name):
    # parsing the street name string for redundant info
    return str(street_name).replace("Rua ", "")


def find_max_jaro_winkler_by_street(df, addr, similarity_ratio):
    # given a df and an address, finds the closest matching street name with at least the given distance value
    largest_similar = -1
    most_similar_row = None
    addr = addr.iloc[0]
    for index, row in df.iterrows():
        if addr["stnd_logr"]:
            # using the jaro winkler formula to see level of similarity: https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance
            cur_jaro_winkler_distance = jaro_winkler(
                stnd_parse_str(row["street"]), stnd_parse_str(addr["stnd_logr"]))
        else:
            cur_jaro_winkler_distance = jaro_winkler(
                stnd_parse_str(row["street"]), stnd_parse_str(addr["logradouro"]))
        if cur_jaro_winkler_distance > largest_similar:
            largest_similar = cur_jaro_winkler_distance
            most_similar_row = row
    if largest_similar > similarity_ratio:
        return most_similar_row
    return None


def fuzzy_match_by_street_name(addr):
    # fuzzy matching the address
    df = perfect_match_by_muni_housenumber(addr).reset_index()
    if df.empty:
        return None
    return find_max_jaro_winkler_by_street(df, addr, .95)


def fuzzy_match_by_street_only(addr):
    # fuzzy matching only the street and using the lines parquet
    df = perfect_match_by_muni(addr).reset_index()
    return find_max_jaro_winkler_by_street(df, addr, .95)


def perform_matches(addr):
    # Process 1
    street_name_match = fuzzy_match_by_street_name(addr)
    if not street_name_match is None:
        return [addr, street_name_match, "by street"]
    # Process 2
    street_only_match = fuzzy_match_by_street_only(addr)
    if not street_only_match is None:
        return [addr, street_only_match, "by street and housenumber"]
    # Both failed (Not similar enough to anything)
    return [addr, None]


def main():
    start = time.perf_counter()
    df = pd.DataFrame()
    df_failed_addresses = pd.DataFrame()
    # number of threads allocated to this python script
    number_of_cores = 4
    with concurrent.futures.ProcessPoolExecutor(max_workers=number_of_cores) as executor:
        results = executor.map(perform_matches, [
            need_to_geocode_df.loc[[i]] for i in range(len(need_to_geocode_df.index))])
        for i, result in enumerate(results):
            df_sent = result[0]
            series_most_similar = result[1]
            if series_most_similar is None:
                # there were not matching streets that were close enough
                df_failed_addresses = pd.concat([df_failed_addresses, df_sent])
                continue
            process = result[2]
            df_most_similar = series_most_similar.to_frame().T
            df = pd.concat([df, df_most_similar.merge(df_sent, how="cross")])
    finish = time.perf_counter()
    # df.to_parquet('addresses_matched.parquet.gzip', compression='gzip')
    # df_failed_addresses.to_parquet(
    #     'df_failed_addresses.parquet.gzip', compression='gzip')
    df.to_csv('addresses_matched.csv', index=False,
              header=True, encoding="utf-8")
    df_failed_addresses.to_csv(
        'failed_addresses.csv', index=False, header=True, encoding="utf-8")
    print(f"This took {round(finish - start, 5)} seconds")


if __name__ == "__main__":
    main()
