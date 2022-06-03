# SyntheticHighways

This is the repository is for the AI Master's Thesis by Kevin Waller, student at the University of Amsterdam, in collaboration with TomTom. It contains all the code necessary to repeat the experiments in the paper, as well as generating the dataset.

## MOD: SyntheticHighways

The dataset is collected from the city-simulation game "Cities: Skylines". A legitimate copy of the game is required to run the mod and collect the data. The mod is written in C# and can be found in the "SyntheticHighways" directory. Parts of the mod is written in Python and Potassco-Clingo, which is responsible for suggesting which changes to make to the map.

## ChangeDetection

The methods which perform change detection, any preprocessing of the dataset after being exported by the mod and any figures found in the paper are generated using the code in the "ChangeDetection" directory. All of the code here is written in Python.
