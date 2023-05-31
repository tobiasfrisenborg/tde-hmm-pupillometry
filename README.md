# Exploring the physiology of transient phase-coupled cortical networks
### The pupillary responses of TDE-HMM states in resting state MEG

**Tobias Frisenborg Christensen**  
201806880@post.au.dk  
MSc Cognitive Science  
1st of June, 2023  
Aarhus University  
School of Communication and Culture  

*View the results at [Streamlit](https://tobiasfrisenborg-tde-hmm-pupillometry-streamlit-z0wd6v.streamlit.app/)*

The project used the trained TDE-HMM solution presented by [Vidaurre et al. (2018)](https://www.nature.com/articles/s41467-018-05316-z)
to classify transient brain states for resting state brain activity collected from 10 subjects.
The subjects had their pupil sizes measured in a dark- and bright room condition.  

The purpose of project was to identify physiological relevance of the HMM states by testing
for signficance between the states and the pupil signal. This was done using a permutation testing
approach.

### Guide to the repository
The results are visualized using streamlit in the `streamlit.py` file.
These results are based on the main data transformations made in the `create_data.ipynb` file.
Most of the data transformations occur in the `utils` files. Perhaps the central one is located
in `utils.data_types.py`. This module contains a python class for structuring the data: the HMM across 
various persistence adjustment levels, the pupil size, etc.
The statistical testing is located in `analysis.ipynb`, which in addition prepares the data for the streamlit.
`manipulate_maps.ipynb` has code for structuring the state maps, preparing for neurosynth, etc. 
Lastly, `neurosynth.ipbynb` is used for scraping results from neurosynth (requires Selenium for circumventing Javascript)
and creating the word clouds based on the meta-anlytic functional decoding analysis.  

*The data is not available beyond the high-level transformations needed for the streamlit.*