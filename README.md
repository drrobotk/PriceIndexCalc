# PriceIndexCalc

Calculate bilateral and multilateral price indices in Python using vectorized methods Pandas or PySpark. These index methods are being used or currently being implemented by many statistical agencies around the world to calculate price indices e.g the Consumer Price Index (CPI). Multilateral methods can use a specified number of time periods to calculate the resulting price index; the number of time-periods used by multilateral methods is commonly defined as a “window length”. 

<img src="https://user-images.githubusercontent.com/51001263/164988385-855ceecf-a5e0-4073-8239-cb1f2304d244.png" width="30%" />

Bilateral methods supported: Carli, Jevons, Dutot, Laspeyres, Paasche, Lowe, geometric Laspeyres, geometric Paasche, Drobish, Marshall-Edgeworth, Palgrave, Fisher, Tornqvist, Walsh, Sato-Vartia, Geary-Khamis, TPD and Rothwell.

Multilateral methods supported: GEKS paired with a bilateral method (e.g GEKS-T aka CCDI), Time Product Dummy (TPD), Time Dummy Hedonic (TDH), Geary-Khamis (GK) method. 

Multilateral methods simultaneously make use of all data over a given time period. The use of multilateral methods for calculating temporal price indices is relatively new internationally, but these methods have been shown to have some desirable properties relative to their bilateral method counterparts, in that they account for new and disappearing products (to remain representative of the market) while also reducing the scale of chain-drift. 

### Directory layout:
    .
    ├── pandas_modules                    # Pandas modules
    │   ├── index_methods.py         
    │   ├── chaining.py
    │   ├── extension_methods.py    # New timeseries extension methods (experimental)                 
    │   ├── helpers.py             
    │   ├── bilateral.py            
    │   ├── multilateral.py
    |   └── weighted_least_squares.py                 
    ├── pyspark_modules                    # PySpark modules (experimental)
    │   ├── index_methods.py              
    │   ├── chaining.py             
    │   ├── helpers.py             
    │   ├── multilateral.py
    |   └── weighted_least_squares.py
    └── README.md
