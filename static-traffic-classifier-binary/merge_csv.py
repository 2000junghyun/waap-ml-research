import os
import csv

def main():
    directory = "csv"
    output_file = "merged-data.csv"

    file_paths = get_csv_file_paths(directory)
    print(f"Found {len(file_paths)} CSV files: {file_paths}")

    if not file_paths:
        print(f"[ERROR] No CSV files in '{directory}' directory.")
        return

    matched, fieldnames = check_fieldnames_match(file_paths)
    if not matched:
        print("Field names do not match.")
        return

    merge_csv_files(file_paths, output_file, fieldnames)


# Directory 내 CSV 파일 목록 확인
def get_csv_file_paths(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]


# 필드명 일치 확인
def check_fieldnames_match(file_paths):
    fieldnames = None
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            else:
                if reader.fieldnames != fieldnames:
                    print(f"[ERROR] Field name mismatch in: {path}")
                    return False, None
    return True, fieldnames


# CSV 파일 통합
def merge_csv_files(file_paths, output_file, fieldnames):
    with open(output_file, 'w', newline='', encoding='utf-8') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        for path in file_paths:
            with open(path, 'r', encoding='utf-8') as in_f:
                reader = csv.DictReader(in_f)
                for row in reader:
                    writer.writerow(row)
    print(f"CSV merge complete: {output_file}")


if __name__ == "__main__":
    main()