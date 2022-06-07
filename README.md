
# UNE COSC301 - Special Topics in Computing
## NFL GPS Game Data Analysis - Spencer Skett

### Project structure

./Project Folder/
    d00_source_data/
        __init__.py
        _the kaggle dataset_ (20 x *.csv)
    d10_code/
        __init__.py
        f10_import_data.py
        f20_filter_data.py
        f30_clean_data.py
        f40_analyse_data.py
        f41_transform_data.py
        f45_state_transitions.py
        f50_visualisations.py
        f51_draw.py
        f52_animate.py
    d20_intermediate_files/
        __init__.py
        _results output_
    d30_results/
        __init__.py
        _results output_ (Final)
    d40_docs/
        __init__.py
    venv/ (Python 3.9.12)
    .gitignore
    main.py
    analyse.py
    LICENSE
    README.md
    testing.ipynb

### Instructions
1. Download the project from Github (_url_) into a suitable root folder and create the folder structure as above
2. Download the Kaggle dataset into the `d00_source_data` folder (https://www.kaggle.com/competitions/nfl-big-data-bowl-2022/data - NOTE: requires account and competition access)
3. Create a virtual environment (tested with Python 3.9.12)
4. Install project dependencies:
   1. pip install numpy pandas matplotlib scipy ray torch sklearn
5. Set options in main.py for data processing
#### WARNING! Steps 5 and 6 have a long runtime and are resource intensive
5. Run main.py
6. Run analyse.py
#### OPTIONAL
7. The three notebooks, `clustering.ipynb`, `density_plots.ipynb` and `route_finder.ipynb` can be used in lieu of `analyse.py` for a more interactive exploration of the data or to focus on a particular aspect. Some additional non-essential functions exist in the notebooks as well 

