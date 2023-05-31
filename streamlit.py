"""
Tobias Frisenborg Christensen, 2023
"""

from pathlib import Path
from PIL import Image

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.mention import mention

from utils.data_io import load_from_pkl

# Data settings
hmm_states = 12
constants = [str(2**i) for i in range(-5, 15+1)]


# Color settings
color_dark = '#48466D'
color_bright = '#3D84A8'

# Function definitions
@st.cache_data
def cached_load_pkl(path):
    return load_from_pkl(path)

@st.cache_data
def cached_load_csv(path):
    return pd.read_csv(path)

@st.cache_data
def state_plot(data, agg_type, constant, stat, ylabel, y_as_pct=False):
    state_df = data.query(f"constant == {constant}")
    
    if stat == 'Mean':
        state_df = pd.DataFrame(
            state_df
            .groupby(['condition', 'state'])[agg_type]
            .mean()
            .reset_index()
        )
    elif stat == 'Median':
        state_df = pd.DataFrame(
            state_df
            .groupby(['condition', 'state'])[agg_type]
            .median()
            .reset_index()
        )
    if stat == 'Min':
        state_df = pd.DataFrame(
            state_df
            .groupby(['condition', 'state'])[agg_type]
            .min()
            .reset_index()
        )
    elif stat == 'Max':
        state_df = pd.DataFrame(
            state_df
            .groupby(['condition', 'state'])[agg_type]
            .max()
            .reset_index()
        )
    elif stat == 'SD':
        state_df = pd.DataFrame(
            state_df
            .groupby(['condition', 'state'])[agg_type]
            .std()
            .reset_index()
        )
    
    fig = px.bar(
        state_df,
        x="state",
        y=agg_type,
        color='condition',
        barmode='group',
        color_discrete_sequence=[color_bright, color_dark],
        title=f"{stat} {agg_type.upper()} by state",
        labels={
            "state": "State",
            agg_type: ylabel,
            "condition": "Condition"
        },
    )
    
    fig.update_layout(
        xaxis=dict(
            tickmode='linear',
            dtick=1)
    )
    
    if y_as_pct:
        fig.update_layout(
            yaxis=dict(tickformat=',.0%')
        )
    
    return fig

@st.cache_data
def constant_plot(data, agg_type, stat, summarize, ylabel, y_as_pct=False):
    constant_df = data
    
    if stat == 'Mean':
        constant_df = pd.DataFrame(
            constant_df
            .groupby(['constant', 'state', 'condition'])[agg_type]
            .mean()
            .reset_index()
        )
    elif stat == 'Median':
        constant_df = pd.DataFrame(
            constant_df
            .groupby(['constant', 'state', 'condition'])[agg_type]
            .median()
            .reset_index()
        )
    elif stat == 'Min':
        constant_df = pd.DataFrame(
            constant_df
            .groupby(['constant', 'state', 'condition'])[agg_type]
            .min()
            .reset_index()
        )
    elif stat == 'Max':
        constant_df = pd.DataFrame(
            constant_df
            .groupby(['constant', 'state', 'condition'])[agg_type]
            .max()
            .reset_index()
        )
    
    
    if summarize == 'Mean':
        constant_df = pd.DataFrame(
            constant_df
            .groupby(['constant', 'condition'])[agg_type]
            .mean()
            .reset_index()
        )
    elif summarize == 'Median':
        constant_df = pd.DataFrame(
            constant_df
            .groupby(['constant', 'condition'])[agg_type]
            .median()
            .reset_index()
        )
    elif summarize == 'Min':
        constant_df = pd.DataFrame(
            constant_df
            .groupby(['constant', 'condition'])[agg_type]
            .min()
            .reset_index()
        )
    elif summarize == 'Max':
        constant_df = pd.DataFrame(
            constant_df
            .groupby(['constant', 'condition'])[agg_type]
            .max()
            .reset_index()
        )
    
    fig = px.line(
        constant_df,
        x="constant",
        y=agg_type,
        color='condition',
        color_discrete_sequence=[color_bright, color_dark],
        title=f"{summarize} of {stat.lower()} state {agg_type.upper()} by state persistence adjustement",
        labels={
            "constant": "Constant (diagonal * 2^x)",
            agg_type: ylabel,
            "condition": "Condition"
        },
    )
    
    if y_as_pct:
        fig.update_layout(
            yaxis=dict(tickformat=',.0%')
        )

    return fig

@st.cache_data
def point_plot(data_states, data_sessions, condition, x, y, title, xlabel, ylabel):
    # Create plot
    fig = go.Figure()

    jitter = .1
    data_sessions['x'] = data_sessions['state'] + np.random.uniform(-jitter, jitter, len(ps_session_df))
    
    fig.add_trace(go.Scatter(
        x=data_sessions['x'],
        y=data_sessions['pupil_size'],
        mode='markers',
        marker=dict(symbol='circle', size=8, color=color_bright if condition == 'Bright room' else color_dark),
        hovertemplate='<b>State</b>: %{x}<br><b>Pupil Size</b>: %{y}<extra></extra>',
    ))
    
    # Horizontal line
    fig.add_trace(go.Scatter(
        x=data_states[x], 
        y=[0] * len(data_states),
        mode='lines',
        line=dict(color='grey', width=1),
        showlegend=False,
        hoverinfo='none',
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(title=xlabel, tickmode='linear', dtick=1),
        yaxis=dict(title=ylabel), showlegend=False
    )
    
    return fig

#@st.cache_data
def heatmap_plot(data, title, xlabel, ylabel, colors='viridis', all_ticks=False):
    # Setup main plot
    fig = px.imshow(
        data,
        title=title,
        color_continuous_scale=colors
    )
    # Adjust labels
    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text=ylabel)
    
    fig.update_layout(yaxis=dict(tickmode='linear', dtick=1))
    
    if all_ticks:
        fig.update_layout(
            xaxis=dict(tickmode='linear', dtick=1),
        )
    
    return fig

def hmm_fit_tab(tab, data, data_type, y_description, y_as_pct):
    with tab:    
        title_dict = {
            'fo': 'Fractional occuppancy',
            'sr': 'Switching rate',
            'dt': 'Dwell time',
        }
        f"##### {title_dict[data_type]} by state"
        f"Explore how {title_dict[data_type].lower()} ({data_type.upper()}) vary between states and across various levels of state persistence adjustments."
        state_param1, state_param2 = st.columns(2)
        constant = state_param1.slider(
            "Select state persistence adjustment (diagonal * 2^x):",
            min_value=-5, max_value=15, value=0, key=f'constant_{data_type}_state')
        stat = state_param2.selectbox(
            "Select a statistic for combining session values",
            ['Mean', 'Median', 'Min', 'Max', 'SD'], index=1, key=f'stat_{data_type}_state')
    
        fig = state_plot(data, data_type, constant, stat, f"{stat} {y_description}", y_as_pct=y_as_pct)
        st.plotly_chart(fig)
    
        f"##### {title_dict[data_type]} across state persistence levels"
        f"Visualize how {title_dict[data_type].lower()} ({data_type.upper()}) is generally affected by state persistence adjustments."
        setting_1, setting_2 = st.columns(2)
        stat = setting_1.selectbox(
            "Select a statistic for combining state values (applied first)",
            ['Mean', 'Median', 'Min', 'Max'], index=1, key=f'stat_{data_type}_constant')
        summarize = setting_2.selectbox(
            f"Select a statistic for combining session values (applied last)",
            ['Mean', 'Median', 'Min', 'Max'], index=3, key=f'summarize_{data_type}_constant')
    
        fig = constant_plot(
            data, data_type, stat, summarize,
            f"{summarize} of state {stat.lower()} {y_description}", y_as_pct=y_as_pct)
        st.plotly_chart(fig)



"""# Exploring the physiology of transient phase-coupled cortical networks
### The pupillary responses of TDE-HMM states in resting state MEG

**Tobias Frisenborg Christensen**  
201806880@post.au.dk  
MSc Cognitive Science  
1st of June, 2023  
Aarhus University  
School of Communication and Culture  """

mention(
    label="Code is available on GitHub",
    icon="github",
    url="https://github.com/tobiasfrisenborg/tde-hmm-pupillometry",
)

"""##### Project description
This dashboard presents the results of my Cognitive Science MSc thesis.
The project used the trained TDE-HMM solution presented by [Vidaurre et al. (2018)](https://www.nature.com/articles/s41467-018-05316-z)
to classify transient brain states for resting state brain activity collected from 10 subjects.
The subjects furthermore had their pupil sizes measured in a dark- and bright room condition.  

The purpose of project was to identify physiological relevance of the HMM states by testing
for signficance between the states and the pupillary response. This was done using a permutation testing
approach.  

The states referred to in the paper are:  
**State 3:** inhibited-pDMN  
**State 4:** excited-pDMN  
**State 11:** inhibited-visual
"""

add_vertical_space(5)


"""## Compare state spectral activity
This section allows you to explore and compare the states obtained using the TDE-HMM.
You can visualize the spectral activity pattern of each state and see a word cloud of the terms
obtained from the meta-analytic functional decoding of the selected spectral maps (see [Neurosynth](https://neurosynth.org/) for more details).
"""
with st.container():
    neurosynth = cached_load_csv('streamlit_data/neurosynth.csv')
    
    freq_maps_dict = {
        "Delta/theta": "fac1",
        "Alpha": "fac2",
        "Beta": "fac3",
        "Wideband": "wideband"
    }
    
    freq_maps_dict_alternate = {
        "Delta/theta": "deltatheta",
        "Alpha": "alpha",
        "Beta": "beta",
        "Wideband": "wideband"
    }
    
    @st.cache_data
    def open_neurosynth_wc(state, freq, background):
        return Image.open(Path(f"streamlit_data/neurosynth/{background}/state{state}_{freq_maps_dict_alternate[freq]}.png"))
    
    @st.cache_data
    def open_brain_map(state, freq, threshold):
        if threshold:
            path = Path(f"streamlit_data/maps_hard_threshold/{state}_{freq_maps_dict_alternate[freq]}.png")
        else:
            path = Path(f"streamlit_data/maps/state_{state}_{freq_maps_dict[freq]}.png")
            
        return Image.open(path)
    
    
    "### Comparison settings"
    "Select which states and frequency bands you want to compare."
    compare_settings_1, compare_settings_2 = st.columns(2)
    
    def compare_state_setings(column, col_n):
        assert col_n in [1, 2]
        state = column.number_input(
            "Select a state",
            min_value=0, max_value=11, value=3 if col_n == 1 else 4, key=f'state_{col_n}')
        freq = column.selectbox(
            "Select frequency band",
            ['Alpha', 'Beta', 'Delta/theta', 'Wideband'], index=3, key=f'freq_{col_n}'
        )
        
        return state, freq
    
    compare_state_1, compare_freq_1 = compare_state_setings(compare_settings_1, 1)
    compare_state_2, compare_freq_2 = compare_state_setings(compare_settings_2, 2)
    

    "### State maps"
    'Visualization of the cortical activity of the selected state and frequency bands.'
    'Note that the "hard thresholds" have been manually defined for a cleaner visualization.'
    map_thresholds = st.checkbox("Use hard thresholds (recommended)", value=True)
    
    # Load and display images
    compare_map_1, compare_map_2 = st.columns(2)
    
    compare_map_1.image(open_brain_map(compare_state_1, compare_freq_1, map_thresholds))
    compare_map_2.image(open_brain_map(compare_state_2, compare_freq_2, map_thresholds))
    
    
    "### Meta-analytic functional decoding"
    "Results of the Neurosynth decoding on spectral state maps. The size of the word in the word cloud is determined by the correlation coefficient of the term."
    compare_wc_settings_1, compare_wc_settings_2 = st.columns(2)
    
    wc_display_type = compare_wc_settings_1.radio(
        "How should data be visualized?", ["Word cloud", "Raw data"], index=0,
    )
    black_bg = compare_wc_settings_2.checkbox("Black background", value=True)
    
    compare_wc_1, compare_wc_2 = st.columns(2)
    
    if wc_display_type == 'Word cloud':
        compare_wc_1.image(open_neurosynth_wc(
            compare_state_1,
            compare_freq_1,
            'black' if black_bg else 'white'
        ))
        
        compare_wc_2.image(open_neurosynth_wc(
            compare_state_2,
            compare_freq_2,
            'black' if black_bg else 'white'
        ))
        
    else:
        def wc_subset_query(state, freq):
            return f"state == {state} & freq == '{freq}'"
        
        compare_wc_1.write(
            neurosynth
            .query(wc_subset_query(compare_state_1, freq_maps_dict_alternate[compare_freq_1]))
            .reset_index()[['term', 'corr']]
        )
        
        compare_wc_2.write(
            neurosynth
            .query(wc_subset_query(compare_state_2, freq_maps_dict_alternate[compare_freq_2]))
            .reset_index()[['term', 'corr']]
        )
add_vertical_space(5)


"""## HMM fit results
This section shows how various aspects of the HMM (based on analysis of the Viterbi path) and the influence of adjusting state persistence.  
"""
fo_tab, sr_tab, dt_tab = st.tabs(["Fractional occupancy", "Switching rate", "Dwell times"])

hmm_fit_tab(fo_tab, cached_load_pkl('streamlit_data/fo_df.pkl'), 'fo', f'FO [%]', True)
hmm_fit_tab(sr_tab, cached_load_pkl('streamlit_data/sr_df.pkl'), 'sr', f"transitions per second", False)
hmm_fit_tab(dt_tab, cached_load_pkl('streamlit_data/dt_df.pkl'), 'dt', f"DT [s]", False)
add_vertical_space(5)


"""## Pupillometry and permutation tests
This section allows the exploration of the pupil sizes, how they vary between states, and the results of the statistical tests.
"""
pupil_size_tab, one_vs_rest_tab, state_pair_tab, dilation_constriction_tab = st.tabs(["Pupil sizes", "One-vs-rest", "State-pairs", "Dilation / constriction"])

with pupil_size_tab:
    # Preprocessing of data
    ps_state_df = load_from_pkl('streamlit_data/ps_state_df.pkl')
    ps_session_df = load_from_pkl('streamlit_data/ps_session_df.pkl')
    
    f"##### Pupil size by state"
    state_param1, state_param2 = st.columns(2)
    condition = state_param1.selectbox(
        "Select condition",
        ['Dark room', 'Bright room'], index=0, key='ps_condition')
    stat = state_param2.selectbox(
        "Select how session values are summarized together",
        ['Mean', 'Median', 'SD'], index=0, key=f'ps_stat')
    state_param2.write("Choose how session values are aggregated together")
    
    ps_state_df = ps_state_df.query(f"condition == '{condition}' & stat == '{stat}'")
    ps_session_df = ps_session_df.query(f"condition == '{condition}' & stat == '{stat}'")
    
    fig = point_plot(ps_state_df, ps_session_df, condition, 'state', 'pupil_size', f"{condition} - {stat} pupil size by state", 'State', f'{stat} pupil size')
    
    st.plotly_chart(fig)

baseline_msg = """##### One-vs-rest test for baseline pupil sizes
Test for differences between baseline pupil size of one state versus the remaining states.
* The test is a permutation test based on 100 permutations
* The p-value represents the proportion of permutations with more extreme values
* p-values are first calculated independently between sessions and then combined using the geometric mean
* The test is calculate in 2 directions (state > rest, state < rest)

You can select multiple different test statistics. For example, when selecting the 'mean' the mean pupil size of the state is compared with the mean pupil size of all the remaining states.
"""
with one_vs_rest_tab:
    baseline_df = cached_load_csv('streamlit_data/ovr_df.csv')
    
    st.write(baseline_msg)
    
    ovr_condition, ovr_stat, ovr_direction = st.columns(3)
    
    condition = ovr_condition.selectbox(
        "Select condition",
        ['Dark room', 'Bright room'], index=0, key='ovr_condition')
    stat = ovr_stat.selectbox(
        "Select statistic",
        ['Mean', 'Median', 'SD', 'Min-Max'], index=0, key='ovr_stat')
    ovr_stat.write(f"*Test statistic used to compare differences between the pupil size for state against remaining states*")
    direction = ovr_direction.radio(
        "Select direction",
        ['State > rest', 'State < rest'], index=0, key='ovr_direction')
    
    baseline_df = (baseline_df
        .loc[
            (baseline_df.condition == condition) & \
            (baseline_df.stat == stat) & \
            ((baseline_df.direction == "Positive") if direction == 'State > rest' else (baseline_df.direction == "Negative")),
            ['state', 'constant', 'p']]
        .pivot(index='state', columns='constant', values='p')
    )

    fig = heatmap_plot(
        baseline_df,
        title=f"{condition} - Baseline ({direction.lower()})",
        xlabel='Constant (diagonal * 2^x)',
        ylabel='State',
        colors='greens',
    )

    st.plotly_chart(fig)
    
svs_msg = """##### State-pair test for baseline pupil sizes
Test for differences in baseline pupil size of state pairs.
* The test is a permutation test based on 100 permutations
* The p-value represents the proportion of permutations with more extreme values
* p-values are first calculated independently between sessions and then combined using the geometric mean

You can select multiple different test statistics. For example, when selecting the 'mean' the mean pupil size of the state 1 is compared to the mean pupil size of state 2.
"""
with state_pair_tab:
    st.write(svs_msg)
    
    # Load data
    svs = {
        'Dark room': {
            'Mean': load_from_pkl("streamlit_data/dark_svs_mean.pkl"),
            'Median': load_from_pkl("streamlit_data/dark_svs_median.pkl"),
        },
        'Bright room': {
            'Mean': load_from_pkl("streamlit_data/bright_svs_mean.pkl"),
            'Median': load_from_pkl("streamlit_data/bright_svs_median.pkl"),
        },
    }
    
    svs_condition, svs_stat, svs_constant = st.columns(3)
    condition = svs_condition.selectbox(
        "Select condition",
        ['Dark room', 'Bright room'], index=0, key='svs_condition')
    stat = svs_stat.selectbox(
        "Select statistic",
        ['Mean', 'Median'], index=0, key='svs_stat')
    svs_stat.write(f"*Test statistic used to compare differences between the pupil sizes of the states*")
    constant = svs_constant.slider(
        "Select state persistence adjustment (diagonal * 2^x):",
        min_value=-5, max_value=15, value=0, key='svs_constant')
    
    fig = heatmap_plot(
        svs[condition][stat][constant],
        title=f"{condition} - State-pair (2^{constant})",
        xlabel='State 1',
        ylabel='State 2',
        colors='reds',
        all_ticks=True,
    )
    
    st.plotly_chart(fig)
    "*Values below the diagonal (lower-left): lower state number < higher state number*"
    "*Values above the diagonal (upper-right): lower state number > higher state number*"

dilation_msg = """##### State-pair test for dilation/constriction
Test for differences between pupil size at state onset and pupil size as state outset.
* The test is a permutation test based on 100 permutations
* The p-value represents the proportion of permutations with more extreme values
* p-values are first calculated independently between sessions and then combined using the geometric mean

You can select multiple different test statistics. For example, when selecting the 'mean' the mean pupil size at state onset is compared to the mean pupil size at state outset.
"""
with dilation_constriction_tab:
    dilation_df = cached_load_csv('streamlit_data/dilation_df.csv')
    
    st.write(dilation_msg)
    
    dilation_condition, dilation_stat, dilation_direction = st.columns(3)
    
    condition = dilation_condition.selectbox(
        "Select condition",
        ['Dark room', 'Bright room'], index=0, key='dilation_condition')
    stat = dilation_stat.selectbox(
        "Select statistic",
        ['Mean', 'Median'], index=0, key='dilation_stat')
    dilation_stat.write(f"*Test statistic used to compare pupil values at state start and pupil values at state end*")
    direction = dilation_direction.radio(
        "Select direction",
        ['Dilation', 'Constriction'], index=0, key='dilation_direction')
    
    dilation_df = (dilation_df
        .loc[
            (dilation_df.condition == condition) & \
            (dilation_df.stat == stat) & \
            ((dilation_df.direction == "Positive") if direction == 'Dilation' else (dilation_df.direction == "Negative")),
            ['state', 'constant', 'p']]
        .pivot(index='state', columns='constant', values='p')
    )
    
    fig = heatmap_plot(
        dilation_df,
        title=f"{condition} - {direction} test",
        xlabel='Constant (diagonal * 2^x)',
        ylabel='State',
        colors='blues',
    )

    st.plotly_chart(fig)
