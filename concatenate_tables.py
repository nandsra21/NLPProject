"""Concatenate two or more tables as data frames.
"""
import argparse
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", nargs="+", help="tables to concatenate")
    parser.add_argument("--separator", default=",", help="separator between columns in the given tables")
    parser.add_argument("--sort-by", help="what to sort the dataframe by (optional)")
    parser.add_argument("--split", type=int, help="if making multiple concatenated df what to split on")
    parser.add_argument("--output", help="concatenated table")

    args = parser.parse_args()
    if args.split is not None:
        i = 0
        composite_list = [args.tables[x:x + args.split] for x in range(0, len(args.tables), args.split)]
        for list_of in composite_list:
            # Concatenate tables.
            df = pd.concat([
                pd.read_csv(table_file, sep=args.separator)
                for table_file in list_of
            ], ignore_index=True)

            df.to_csv(args.output[i], sep=args.separator, header=True, index=False)
            i = i + 1
    else:
        # Concatenate tables.
        df = pd.concat([
            pd.read_csv(table_file, sep=args.separator)
            for table_file in args.tables
        ], ignore_index=True)

        if args.sort_by is not None:
            df = df.sort_values(by=[args.sort_by])
            cols_to_order = [args.sort_by]
            new_columns = cols_to_order + (df.columns.drop(cols_to_order).tolist())
            df = df[new_columns]

        df.to_csv(args.output, sep=args.separator, header=True, index=False)
