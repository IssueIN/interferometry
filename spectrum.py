import csv
import matplotlib.pyplot as plt

x = []
y = []

file_path = 'data/hg_spectrum.csv'

with open(file_path, 'r') as csvfile:
    # Create a CSV reader object
    csvreader = csv.reader(csvfile)
    
    # Skip lines until the [Data] section is found
    for row in csvreader:
        if row and row[0].strip() == '[Data]':
            break  # Found the data section, exit the loop
    
    # Read the rest of the data
    for row in csvreader:
        # Make sure the row has at least two elements
        if len(row) >= 2:
            try:
                # Convert the string data to the appropriate type (float in this case)
                x_value = float(row[0])
                y_value = float(row[1])
                x.append(x_value)
                y.append(y_value)
            except ValueError:
                # Handle the error if the data cannot be converted to float
                print(f"Could not convert data to float: {row}")
        else:
            # Handle the error if a row doesn't have enough columns
            print(f"Row has insufficient columns: {row}")

# Now x and y contain the data points as floats

# x = [float(i) for i in x]
# y = [float(j) for j in y]

plt.plot(x, y)
plt.show()