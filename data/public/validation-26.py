##
# VALIDATION SCRIPT FOR THE TERMINOLOGY SHARED TASK AT WMT2026
# INSTRUCTIONS:
# 0. HAVE PYTHON3 INSTALLED ON YOUR MACHINE; NO CUSTOM PACKAGES ARE REQUIRED
# 1. VALIDATE EACH TRACK (MT WITH TERMINOLOGIES OR MT WITH SAMPLE BITEXTS) SEPARATELY
# 2. STORE ALL FILES OF YOUR SUMBISSION FOR A PARTICULAR TRACK IN A SINGLE (NON-NESTED) DIRECTORY, e.g. /your_submission/track1/submissions
# 3. STORE ALL INPUT FILES OF THIS TRACK IN ANOTHER (NON-NESTED) DIRECTORY, e.g. /working_directory/track1/inputs
# 4. RUN THE CODE WITH THE FOLLOWING PARAMETERS:
#       -t OR --track: CHOOSE 1 FOR EXPLICIT TERMINOLOGIES OR 2 FOR SAMPLE BITEXTS
#       -i OR --inputs: PATH TO YOUR FOLDER WITH INPUT FILES
#       -o OR --outputs: PATH TO THE FOLDER WITH YOUR SUBMITTED FILES

# USAGE:
# FOR TRACK 1 (DOC-LEVEL MT WITH EXPLICIT TERMINOLOGIES):
# python validation-26.py -t 1 -i /working_directory/track1/inputs -o /your_submission/track1/outputs
# FOR TRACK (DOC-LEVEL MT WITH SAMPLE BITEXTS):
# python validation-26.py -t 2 -i /working_directory/track2/inputs -o /your_submission/track2/outputs

# EXAMPLE OF THE OUTPUT:
# IF EVERYTHING IS FINE, YOU WILL SEE THE FOLLOWING LINES:

# Test 1/2: checking file structure, naming validation, and file format...
# FORMAT CHECK: DONE
# Test 2/2: checking consistency with source data...
# CHECK CONSISTENCY WITH SOURCE DATA: DONE
# All entries formated correctly. The files are ready for submission.

# OTHERWISE, YOU WILL SEE THE CORRESPONDING ERRORS POINTING AT THE FORMATTING PROBLEMS, FOR INSTANCE:
# ERROR: bad.random.energy.eseu.json missing in submission
# ERROR: inconsistent number of lines: bad.proper.energy.eseu.json: 5 VS text.energy.eseu.json: 12

import json
import os
import argparse

def _naming_check(folder, track):
    """
    Validates file naming consistency.

    Checks:
    - All files follow the pattern: {system}.{mode}.{domain}.{pair}.json
    - System name is consistent across all files
    - Language pair to domain mapping is valid
    - All required translation modes are present for each domain-pair combination
    - All domains for a given pair are present

    :param folder: str, path to submission folder
    :param track: int, track number (1 or 2)
    :raises AssertionError: if naming checks fail
    """
    files = os.listdir(folder)
    json_files = [f for f in files if f.endswith('.json')]
    if not json_files:
        raise AssertionError("ERROR: no JSON files found in submission")

    # Define valid domain-pair mappings
    valid_mappings = {
        "eseu": {"1": ["energy", "automotion"], "2": ["machine-tool", "railways"]},
        "enpl": {"1": ["mechanical-engineering", "medical"], "2": ["mechanical-engineering", "medical"]},
        "zhen": {"2": ["finance"]}
    }

    # Check naming format and extract components
    parsed_files = {}
    system_names = set()

    for fname in json_files:
        parts = fname.split('.')

        # Check naming pattern: {system}.{mode}.{domain}.{pair}.json
        if len(parts) != 5 or parts[-1] != 'json':
            raise AssertionError(f"ERROR: inconsistent file naming: {fname}.")

        system, mode, domain, pair = parts[0], parts[1], parts[2], parts[3]

        # Validate mode
        if track == 1 and mode not in ["noterm", "proper", "random"]:
            raise AssertionError(f"ERROR: inconsistent file naming: {fname}.")
        elif track == 2 and mode not in ["noterm", "sample"]:
            raise AssertionError(f"ERROR: inconsistent file naming: {fname}.")

        system_names.add(system)

        # Validate language pair to domain mapping
        if pair not in valid_mappings:
            raise AssertionError(f"ERROR: inconsistent language pair to domain mapping: {fname} (check if it's the right track)")

        valid_domains = valid_mappings[pair].get(str(track), [])
        if domain not in valid_domains:
            raise AssertionError(f"ERROR: inconsistent language pair to domain mapping: {fname} (check if it's the right track)")

        # Store parsed file info
        key = (domain, pair)
        if key not in parsed_files:
            parsed_files[key] = set()
        parsed_files[key].add(mode)

    # Check system name consistency
    if len(system_names) != 1:
        raise AssertionError(f"ERROR: inconsistent system naming: {', '.join(system_names)}")

    # Check that all required modes are present for each domain-pair combination
    for (domain, pair), modes in parsed_files.items():
        if track == 1:
            required_modes = {"noterm", "proper", "random"}
        elif track == 2:
            required_modes = {"noterm", "sample"}

        if modes != required_modes:
            raise AssertionError(f"ERROR: not all translation modes are submitted: {domain} x {pair}")

    # Check that all domains for each pair are present
    for pair in ["eseu", "enpl", "zhen"]:
        if pair not in valid_mappings:
            continue

        valid_domains = valid_mappings[pair].get(str(track), [])
        if not valid_domains:
            continue

        # Check which domains appear in parsed files
        submitted_domains = {domain for (d, p), _ in parsed_files.items() if p == pair for domain in [d]}
        required_domains = set(valid_domains)

        if required_domains and submitted_domains and submitted_domains != required_domains:
            raise AssertionError(f"ERROR: not all domains for {pair} are submitted")


def _file_content_check(folder):
    """
    Validates file content format and structure.

    Checks:
    - All files are valid JSON
    - Files contain List[Str] data structure

    :param folder: str, path to submission folder
    :raises AssertionError: if content checks fail
    """
    files = os.listdir(folder)
    json_files = [f for f in files if f.endswith('.json')]

    for fname in json_files:
        filepath = os.path.join(folder, fname)

        # Check if file is valid JSON
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise AssertionError(f"ERROR: {fname} is not JSON-valid")

        # Check if data structure is List[Str]
        if not isinstance(data, list):
            raise AssertionError(f"ERROR: {fname} should be List[Str] data structure")

        for item in data:
            if not isinstance(item, str):
                raise AssertionError(f"ERROR: {fname} should be List[Str] data structure")


def format_check(folder, track):
    """
    Performs format and naming validation on submission files.

    Executes two validation checks:
    1. Naming consistency (_naming_check)
    2. File content validity (_file_content_check)

    :param folder: str, path to submission folder
    :param track: int, track number (1 or 2)
    :return: str, confirmation message if all checks pass
    :raises AssertionError: if any validation fails
    """
    print("Test 1/2: checking file structure, naming validation, and file format...")

    _naming_check(folder, track)
    _file_content_check(folder)

    return "FORMAT CHECK: DONE"


def content_check(submission_folder, input_folder):
    """
    Validates content consistency between submission and input files.

    Checks:
    - Number of strings in submission and input files match
    - Number of paragraphs in each string match between submission and input

    :param submission_folder: str, path to submission files folder
    :param input_folder: str, path to system inputs folder
    :param track: int, track number (1 or 2)
    :raises AssertionError: if content checks fail
    """
    print("Test 2/2: checking consistency with source data...")
    files = os.listdir(submission_folder)
    json_files = [f for f in files if f.endswith('.json')]
    #print(json_files, len(json_files))
    for fname in json_files:
    #    print(fname)
        # Parse filename: {system}.{mode}.{domain}.{pair}.json
        parts = fname.split('.')
        if len(parts) != 5:
            continue

        domain = parts[2]
        pair = parts[3]

        # Find corresponding input file: text.{domain}.{pair}.json
        input_fname = f"text.{domain}.{pair}.json"
        input_filepath = os.path.join(input_folder, input_fname)
        submission_filepath = os.path.join(submission_folder, fname)

        # Check if input file exists
        if not os.path.exists(input_filepath):
            raise AssertionError(f"ERROR: input file not found: {input_fname}")

        # Load both files
        try:
            with open(submission_filepath, 'r', encoding='utf-8') as f:
                submission_data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise AssertionError(f"ERROR: {fname} is not valid JSON")

        try:
            with open(input_filepath, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise AssertionError(f"ERROR: {input_fname} is not valid JSON")
    #    print(f'submission_data: {len(submission_data)}, input_data: {len(input_data)}')

        # Check 1: Number of strings must match
        if len(submission_data) != len(input_data):
            raise AssertionError(
                f"ERROR: inconsistent number of lines: {submission_filepath}: "
                f"{len(submission_data)} VS {input_filepath}: {len(input_data)}"
            )

        # Check 2: Compare paragraph counts for each string
        for string_id, (submission_string, input_string) in enumerate(
                zip(submission_data, input_data)
        ):
            # Determine the delimiter: try \n\n first, then \n
            # Count paragraphs in submission
            if '\n\n' in submission_string:
                submission_paragraphs = submission_string.split('\n\n')
            else:
                submission_paragraphs = submission_string.split('\n')

            # Count paragraphs in input
            if '\n\n' in input_string:
                input_paragraphs = input_string.split('\n\n')
            else:
                input_paragraphs = input_string.split('\n')

            # Check if paragraph counts match
            if len(submission_paragraphs) != len(input_paragraphs):
                raise AssertionError(
                    f"ERROR: inconsistent number of paragraphs: "
                    f"{submission_filepath}: line {string_id}: "
                    f"{len(submission_paragraphs)} VS {input_filepath}: "
                    f"line {string_id}: {len(input_paragraphs)}"
                )

    return "CHECK CONSISTENCY WITH SOURCE DATA: DONE"


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-t", "--track", type=int, help="competition track: 1 for document-level MT with terminologies, 2 for document-level MT with sample bitexts")
    argparser.add_argument("-i", "--inputs", type=str, help="file path for input folder given track (no internal folders!)")
    argparser.add_argument("-o", "--outputs", type=str, help="file path for input folder for given track (no internal folders, all files for a given system)")
    args = argparser.parse_args()

    if args.track == 1:
        pass
    elif args.track == 2:
        pass
    else:
        raise Exception("you must choose a track: 1 document-level MT with terminologies, 2 for document-level MT with sample bitexts")


    # 1. checking the file assortiment and formats
    format_check_message = format_check(args.outputs, track=args.track)
    print(format_check_message)

    # 2. checking the consistency with the input files
    content_check_message = content_check(args.outputs, args.inputs)
    print(content_check_message)

    print ("All entries formated correctly. The files are ready for submission.")

if __name__ == "__main__":
    main()