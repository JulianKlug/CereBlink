import os
import pandas as pd
import io
import getpass
import msoffcrypto
import asyncio

def load_encrypted_xlsx(file_path, sheet_name=None, password=None):
    """
        Loads an encrypted Excel file and returns the data as a pandas DataFrame.

        Args:
            file_path (str): The path to the encrypted Excel file.
            sheet_name (str, optional): The name of the sheet to load. Defaults to None.

        Returns:
            pandas.DataFrame: The data from the specified sheet as a DataFrame.
    """
    if password is None:
        password = getpass.getpass()
    decrypted_workbook = io.BytesIO()
    with open(file_path, 'rb') as file:
        office_file = msoffcrypto.OfficeFile(file)
        office_file.load_key(password=password)
        office_file.decrypt(decrypted_workbook)

    if sheet_name is None:
        df = pd.read_excel(decrypted_workbook)
    else:
        df = pd.read_excel(decrypted_workbook, sheet_name=sheet_name)
    return df


def flatten(l):
    return [item for sublist in l for item in sublist]


def safe_conversion_to_datetime(date):
    try:
        return pd.to_datetime(date)
    except:
        return pd.NaT


def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped