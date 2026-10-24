# Empirical data

The empirical application uses two CSV files exported from Wind Information:

- `raw/16PE-DY.csv`: price/earnings, dividend yield, and stock returns.
- `raw/16PS-PB.csv`: price/sales, price/book, and stock returns.

The files contain monthly observations for 16 economies. Each file has a date column followed by three blocks of 16 columns. The first data row identifies the equity indices.

The scripts construct the valuation variables and estimation samples from these files.
