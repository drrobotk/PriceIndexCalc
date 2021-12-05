# IndexNumCalc

Calculate bilateral and multilateral price indices, the latter using the GEKS paired with a bilateral method (e.g GEKS-T aka CCDI), Time Product Dummy (TPD), Time Dummy Hedonic (TDH), Geary-Khamis (GK) method. 

Multilateral methods simultaneously make use of all data over a given time period. The use of multilateral methods for calculating temporal price indices is relatively new internationally, but these methods have been shown to have some desirable properties relative to their bilateral method counterparts, in that they account for new and disappearing products (to remain representative of the market) while also reducing the scale of chain-drift. They are used or currently being implemented by many statistical agencies around the world to calculate price indices e.g the Consumer Price Index (CPI).

Multilateral methods can use a specified number of time periods to calculate the resulting price index; the number of time-periods used by multilateral methods is commonly defined as a “window length”. Currently we use the entire timeseries length as the window length until timeseries extension methods are to be implemented.

Directory layout:
    .
    ├── ...
    ├── README.md
    ├── pandas                    # Pandas modules
    │   ├── index_methods.py         
    │   ├── chaining.py             
    │   ├── helpers.py             
    │   ├── bilateral.py            
    │   ├── multilateral.py
    |   └── wls.py                 
    ├── pyspark                    # PySpark modules
    │   ├── index_methods.py              
    │   ├── chaining.py             
    │   ├── helpers.py             
    │   ├── multilateral.py
    |   └── wls.py
    └── ...  