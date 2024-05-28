import csv

# Replace 'input.csv' and 'output.tsv' with your file names
BASE_PATH = './kg-datasets/RedHatFineTune/raw'

files = ['train.csv', 'test.csv', 'valid.csv']

for input_file in files:

    input_file = BASE_PATH + '/' + input_file
    output_file = input_file.replace('.csv', '.txt')
    
    # Open the CSV file for reading
    with open(input_file, mode='r', encoding='utf-8') as csv_file:
        # Create a CSV reader
        csv_reader = csv.reader(csv_file)

        # Open the TSV file for writing
        with open(output_file, mode='w', encoding='utf-8', newline='') as tsv_file:
            # Create a CSV writer with tab delimiter
            tsv_writer = csv.writer(tsv_file, delimiter='\t')

            # ignore the first row of the CSV file for the headers
            next(csv_reader)

            # Write each row from the CSV to the TSV
            for row in csv_reader:
                tsv_writer.writerow(row)
