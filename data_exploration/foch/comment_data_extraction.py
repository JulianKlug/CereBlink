import os
import pandas as pd
import re
from fpdf import FPDF


def sql_to_pd_df(file_path):
    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        data = file.read()

    # Regular expression pattern to match each insert statement
    pattern = r'Insert into EXPORT_TABLE.*?values \((\d+),to_timestamp\(\'([^\']+)\',\'[^\']*\'\),\'((?:[^\']|\'\')*)\',\s*(EMPTY_CLOB\(\)|(?:TO_CLOB\(q\'\[(.*?)\'\)(?:\s*\|\|\s*TO_CLOB\(q\'\[(.*?)\'\))*)|\'(.*?)\')\s*\)'

    # Find all matches in the file content
    matches = re.findall(pattern, data, re.DOTALL)

    # Lists to store the extracted information
    patient_ids = []
    datetimes = []
    headers = []
    notes = []

    # Loop over the matches and store the extracted data
    for match in matches:
        patient_id = match[0]
        datetime = match[1]
        header = match[2].replace("''", "'")  # Replace double single quotes with a single quote for headers

        # Handle EMPTY_CLOB case
        if match[3] == "EMPTY_CLOB()":
            note = ""  # No notes for EMPTY_CLOB case
        else:
            # If the note is provided directly as a string
            if match[6]:  # Regular string notes (non-TO_CLOB)
                note = match[6]
            else:
                # Otherwise, we are dealing with TO_CLOB concatenations
                first_note = match[4] if match[4] else ""
                additional_clobs = re.findall(r'TO_CLOB\(q\'\[(.*?)\'\)', match[3], re.DOTALL)[1:]
                # Combine all parts including the first TO_CLOB block and subsequent blocks
                all_parts = [first_note] + additional_clobs
                note = " ".join(filter(None, all_parts))

        patient_ids.append(patient_id)
        datetimes.append(datetime)
        headers.append(header)
        notes.append(note)

    # Create a DataFrame to organize the data
    data_dict = {
        "PatientID": patient_ids,
        "Datetime": datetimes,
        "Header": headers,
        "Notes": notes
    }

    df = pd.DataFrame(data_dict)
    return df


def test_sql_to_pd_df(path):
    df = sql_to_pd_df(path)

    with open(path, 'r', encoding='ISO-8859-1') as file:
        data = file.read()

    insert_count = data.count("Insert into EXPORT_TABLE")

    assert len(df) == insert_count, f"Expected {insert_count} rows, but got {len(df)} rows"

    # Regular expression pattern to match the patient ID and timestamp
    pattern = r'values\s*\((\d+),to_timestamp\(\'([\d-]+\s[\d:.]+)'

    # Find all occurrences of the pattern
    matches = re.findall(pattern, data)

    # Create a DataFrame from the extracted matches
    td_df = pd.DataFrame(matches, columns=['PatientID', 'Datetime'])

    # Check if the extracted data matches the expected data
    diff = pd.concat([td_df, df[['PatientID', 'Datetime']]]).drop_duplicates(keep=False)

    assert len(diff) == 0, f"Data mismatch found in {len(diff)} rows"


# Function to generate a PDF for each patient
def create_patient_pdf(patient_id, datetime_list, header_list, notes_list, output_dir):
    # Create a PDF object
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Add a page
    pdf.add_page()

    # Set title with a standard macOS font (Arial)
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(200, 10, f'Patient ID: {patient_id}', ln=True, align='C')

    # Add notes for each datetime, header, and note
    for dt, header, note in zip(datetime_list, header_list, notes_list):
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(200, 10, f'Datetime: {dt}', ln=True)
        pdf.cell(200, 10, f'Header: {header}', ln=True)
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 10, f'Notes: {note}')
        pdf.ln(10)  # Add a line break between notes

    # Save the PDF to a file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create directory if not exists
    file_path = f"{output_dir}/patient_notes_{patient_id}.pdf"
    pdf.output(file_path)
    return file_path


# Group data by patient ID
def generate_pdfs_for_all_patients(df, output_dir):
    # Group data by patient ID
    patient_data = {}

    for idx, row in df.iterrows():
        patient_id = row['PatientID']
        if patient_id not in patient_data:
            patient_data[patient_id] = {"datetimes": [], "headers": [], "notes": []}
        patient_data[patient_id]["datetimes"].append(row['Datetime'])
        patient_data[patient_id]["headers"].append(row['Header'])
        patient_data[patient_id]["notes"].append(row['Notes'])

    # Generate PDFs for each patient
    pdf_files = []
    for patient_id, data in patient_data.items():
        pdf_file = create_patient_pdf(patient_id, data["datetimes"], data["headers"], data["notes"], output_dir)
        pdf_files.append(pdf_file)

    return pdf_files


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Extract data from SQL file and generate PDFs for each patient.')
    parser.add_argument('-s', '--sql_file', type=str, required=True, help='Path to the SQL file')
    parser.add_argument('-o', '--output_dir', type=str, default='output_pdfs', help='Output directory for PDF files')
    args = parser.parse_args()

    # Test the function
    test_sql_to_pd_df(args.sql_file)

    # Extract data from SQL file
    df = sql_to_pd_df(args.sql_file)

    # Generate PDFs for each patient
    pdf_files = generate_pdfs_for_all_patients(df, args.output_dir)
