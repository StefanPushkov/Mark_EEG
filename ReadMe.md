`data` folder - store raw EEG data in .txt format

1. run `main_preparation.py` inside `DataPreparation` folder, it creates file with processed raw data for classifier fitting
2. run `RandomForest_main.py` inside `RandomForest` folder, it fit classifier and save it inside `models` folder

classes: 0 = кнопка не нажата 1 = кнопка нажата левой 2 = кнопка нажата правой


# Plots # 
1. Average precision score plot
2. Extension of Precision-Recall curve to multi-class plot 

### Color - Class ###
For Extension of Precision-Recall curve to multi-class plot 
- Class 0 - Navy color
- Class 1 - Turquoise color
- Class 2 - Darkorange color
- Average precision score line - Gold color