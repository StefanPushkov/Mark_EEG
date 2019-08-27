from DataPreparation.txt_to_csv import text_to_csv
from DataPreparation.auto_markup import auto_markup
from DataPreparation.to_txt import convert_to_txt
from DataPreparation.hex_to_dec import hex_to_dec


# Function txt_to_csv returns '../converted_data/{0}CSV.csv'.format(num)
csv29 = text_to_csv('OBCI_29_SucksAssFull.TXT', 29)


# Function auto_markup returns "../converted_data/markup_class{0}.csv".format(num)
marked_up29 = auto_markup(csv29)


# Function convert_to_txt returns '../converted_data/full_prepared{0}.txt'.format(num)
txt29 = convert_to_txt(marked_up29)


# Function hex_to_dec returns 'decimal{0}.csv'.format(num)
dec29 = hex_to_dec(txt29)


# Merging to one
