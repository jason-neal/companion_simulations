"""Script to load part files into a sql database."""
import argparse
import glob as glob
import sys

import pandas as pd
from sqlalchemy import create_engine


def _parser():
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='Chisquare analysis.')
    parser.add_argument('database', help='database')
    # parser.add_argument('-s', '--suffix', help='Suffix to add to database name.')
    # parser.add_argument('-v', '--verbose', help='Turn on Verbose.', action="store_true")

    return parser.parse_args()


def main(database):

    csv_database = create_engine('sqlite:///{}.db'.format(database), echo=True)

    df = pd.read_sql_query('SELECT * FROM table', csv_database)

    df.plot()
    plt.show()


if __name__ == '__main__':
    args = vars(_parser())
    opts = {k: args[k] for k in args}

    sys.exit(main(**opts))
