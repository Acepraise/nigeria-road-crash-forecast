# Data quality report

Source: Federal Road Safety Corps (FRSC) state crash returns, 2020Q4 to 2024Q2.
Rows written: 555 (37 states including FCT x 15 quarters).

## Issues found

- Blocks 16 to 19 of the sheet are yearly totals (2021, 2022, 2023, H1 2024). Each matches the sum of its quarters exactly, so they were dropped.
- Borno 2021Q3: blank killed, total_casualties in the source.
  - Recovered Borno 2021Q3 killed = 18 from the national total minus all other states.
- Benue 2021Q3: total casualties 113 does not equal injured + killed (162). Casualties are recomputed as injured + killed; the model does not use this column.
