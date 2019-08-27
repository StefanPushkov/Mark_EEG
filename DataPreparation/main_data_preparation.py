from DataPreparation.txt2csv import txt2csv
from markup_functions.auto_markup import auto_markup
from DataPreparation.to_txt import convert_to_txt
from converters.hex2dec import hex2dec
from converters.merge_to_one_csv import merging
from config import *

# Convert EEG raw data from txt format to csv
# function txt2csv returns 'converted_data/{0}CSV.csv'

csv29 = txt2csv(raw_data+"OBCI_29_SucksAssFull.txt", 29)
csv25 = txt2csv(raw_data+"OBCI_25_SUCKSESSFULL.txt", 10)
csv26 = txt2csv(raw_data+"OBCI_26_SUCKSESSFULL.txt", 20)
csv27 = txt2csv(raw_data+"OBCI_27_SUCKSESSFULL.txt", 30)

# Auto markup classes, creates new file("markup_classXX.csv") in prepared_data folder
# function returns 'prepared_data/markup_class{0}.csv'
csv_mark29 = auto_markup(csv29)
csv_mark25 = auto_markup(csv25)
csv_mark26 = auto_markup(csv26)
csv_mark27 = auto_markup(csv27)

# convert_to_txt returns '../converted_data/full_prepared{0}.txt'
txt_final29 = convert_to_txt(csv_mark29)
txt_final25 = convert_to_txt(csv_mark25)
txt_final26 = convert_to_txt(csv_mark26)
txt_final27 = convert_to_txt(csv_mark27)


# hex2dec returns "decimal{0}.csv"
dec29 = hex2dec(txt_final29)
dec25 = hex2dec(txt_final25)
dec26 = hex2dec(txt_final26)
dec27 = hex2dec(txt_final27)

# Returns
full_csv = merging(file1=dec25, file2=dec26, file3=dec27, out_file="final_merged_Kate")

