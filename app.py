import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(page_title="Traitors Analytics", layout="wide")


st.markdown("""
    <style>
    /* App background & text */
    .stApp { background-color: #f7f7f7; color: #111111; font-family: 'Segoe UI', sans-serif; }
    
    /* Metric styling */
    div[data-testid="stMetricValue"] { font-size: 1.8rem; color: #e63946; font-weight: bold; }

    /* Headers */
    h2, h3 { color: #1d3557; }

    /* Chart container for padding & rounded corners */
    .chart-container { 
        background-color: #ffffff; 
        padding: 1rem; 
        border-radius: 12px; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.1); 
        margin-bottom: 1.5rem;
    }

    /* Control container styling */
    .control-container {
        background-color: #ffffff; 
        padding: 1rem; 
        border-radius: 12px; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.05); 
        margin-bottom: 2rem;
    }

    /* Divider spacing */
    .stDivider { margin: 2rem 0; }
    </style>
""", unsafe_allow_html=True)

AGE_ORDER = ["<30", "30-44", "45-59", "60+"]

COLOR_MAP = {
    "white": "#f1690e",             
    "person_of_color": "#ffdf61",   
    "female": "#e377c2",            
    "male": "#1f77b4",              
    "<30": "#0068c9",
    "30-44": "#83c9ff",
    "45-59": "#ffabab",
    "60+": "#ff2b2b",
}


# @st.cache_data
def load_all_data():
    age = pd.read_csv('outputs/age_survival_stats.csv')
    base = pd.read_csv('outputs/baseline_composition.csv')
    early = pd.read_csv('outputs/early_banishment_stats.csv')
    surv = pd.read_csv('outputs/survival_stats.csv')
    finalist = pd.read_csv('outputs/finalist_composition.csv')
    votes = pd.read_csv('outputs/early_vote_composition.csv')
    baseline = pd.read_csv('outputs/baseline_rounds.csv')
    return age, base, early, surv, finalist, votes, baseline

age_df, base_df, early_df, surv_df, finalist_df, votes_df, baseline_df = load_all_data()

st.title("The Traitors: Data Analysis ")
st.info("Select version and seasons to aggregate data. Choosing multiple will sum the counts and recalculate averages.  \n"
        "Note: At this time, only the UK version is available. C* indicates Celebrity season *.")

available_versions = ["UK"]
selected_versions = st.multiselect(
    "Select Show Versions", 
    options=available_versions, 
    default=available_versions
)

available_seasons = sorted([s for s in age_df['season'].unique() if s != 'all'], key=lambda x: str(x))
selected_seasons = st.multiselect(
    "Select Seasons", 
    options=available_seasons, 
    default=available_seasons
)

if not selected_seasons:
    st.error("Please select at least one season.")
    st.stop()

def filter_s(df):
    return df[df['season'].astype(str).isin([str(s) for s in selected_seasons])]

f_age = filter_s(age_df)
f_base = filter_s(base_df)
f_early = filter_s(early_df)
f_surv = filter_s(surv_df)
# f_votes = filter_s(votes_df)


tab1, tab2, tab3, tab4 = st.tabs(["Early Banishments", "Survival Time", "Age Analysis", "Voting Patterns"])

with tab1:
    st.header("Baseline vs. Early Banishment")
    st.write("Compare the cast composition at the start vs. who was banished in the first 4 episodes.")
    
    col1, col2 = st.columns(2)
   
    # Race Section
    with col1:
        st.subheader("Race Impact")
        eth_base = f_base[f_base['group_type'] == 'ethnicity_group'].groupby('group_value')['proportion'].mean().reset_index()
        eth_early = f_early[f_early['group_type'] == 'ethnicity_group'].groupby('group_value')['early_banished'].sum().reset_index()
        
        # Comparison Chart
        fig_eth = go.Figure()
        fig_eth.add_trace(go.Pie(labels=eth_base['group_value'], values=eth_base['proportion'], 
                                 name="Baseline", hole=0.6, domain={'x': [0, 0.45]}, title="Starting Cast"))
        fig_eth.add_trace(go.Pie(labels=eth_early['group_value'], values=eth_early['early_banished'], 
                                 name="Early Banished", hole=0.6, domain={'x': [0.55, 1]}, title="Early Banished"))
        st.plotly_chart(fig_eth, use_container_width=True)

    # Gender Section
    with col2:
        st.subheader("Gender Impact")
        gen_base = f_base[f_base['group_type'] == 'Inferred_Gender'].groupby('group_value')['proportion'].mean().reset_index()
        gen_early = f_early[f_early['group_type'] == 'Inferred_Gender'].groupby('group_value')['early_banished'].sum().reset_index()
        
        fig_gen = go.Figure()
        fig_gen.add_trace(go.Pie(labels=gen_base['group_value'], values=gen_base['proportion'], 
                                 hole=0.6, domain={'x': [0, 0.45]}, title="Starting Cast", marker_colors=['#e377c2', '#1f77b4']))
        fig_gen.add_trace(go.Pie(labels=gen_early['group_value'], values=gen_early['early_banished'], 
                                 hole=0.6, domain={'x': [0.55, 1]}, title="Early Banished",
                                 marker_colors=['#e377c2', '#1f77b4']))
        st.plotly_chart(fig_gen, use_container_width=True)


with tab2:
    st.header("Survival Longevity and Finalists")
    st.write("Average episodes survived by intersectional groups and finalists composition.")

    f_surv['total_ep'] = f_surv['mean_episode'] * f_surv['count']
    surv_grouped = (
        f_surv
        .groupby(['Inferred_Gender', 'ethnicity_group'])
        .agg({'total_ep': 'sum', 'count': 'sum'})
        .reset_index()
    )
    surv_grouped['avg_survival'] = surv_grouped['total_ep'] / surv_grouped['count']

    fig_surv = px.bar(
        surv_grouped,
        x='Inferred_Gender',
        y='avg_survival',
        color='ethnicity_group',
        barmode='group',
        labels={'avg_survival': 'Average Episodes Survived'},
        title= "Survival Longevity",
        color_discrete_map=COLOR_MAP
    )
    
    f_finalists = finalist_df[
        (finalist_df['season'].astype(str).isin([str(s) for s in selected_seasons])) &
        (finalist_df['season'] != 'all')
    ]
    gender_final = f_finalists[f_finalists['group_type'] == 'Inferred_Gender']
    race_final = f_finalists[f_finalists['group_type'] == 'ethnicity_group']

    fig_fg = px.pie(
        gender_final,
        names='group_value',
        values='proportion',
        hole=0.5,
        color='group_value',
        color_discrete_map=COLOR_MAP,
        title="Finalists by Gender"
    )

    fig_fr = px.pie(
        race_final,
        names='group_value',
        values='proportion',
        hole=0.5,
        color='group_value',
        color_discrete_map=COLOR_MAP,
        title="Finalists by Race"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.plotly_chart(fig_surv, use_container_width=True, key="surv")

    with col2:
        st.plotly_chart(fig_fg, use_container_width=True, key="finalist_gender")

    with col3:
        st.plotly_chart(fig_fr, use_container_width=True, key="finalist_race")


with tab3:
    st.header("Age Group Statistics")
    f_age['age_group'] = pd.Categorical(f_age['age_group'], categories=AGE_ORDER, ordered=True)
    age_agg = f_age.groupby('age_group')['count'].sum().reset_index()
    age_agg = age_agg.sort_values('age_group') # This ensures <30 is first
    
    col_a, col_b = st.columns([1, 2])
    with col_a:
        fig_age_p = px.pie(age_agg, values='count', names='age_group', hole=0.4, color_discrete_map=COLOR_MAP)
        fig_age_p.update_layout(
            annotations=[
                dict(
                    text="<b>Starting Cast</b>",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=12, color="#818181"),
                    align="center"
                )
            ]
        )
        st.plotly_chart(fig_age_p, use_container_width=True)
    with col_b:
        # Re-calculate survival for age
        f_age['total_ep'] = f_age['mean_episode'] * f_age['count']
        age_surv = f_age.groupby('age_group').agg({'total_ep':'sum', 'count':'sum'}).reset_index()
        age_surv['avg_survival'] = age_surv['total_ep'] / age_surv['count']
        fig_age_b = px.bar(age_surv, x='age_group', y='avg_survival', color='age_group', color_discrete_map=COLOR_MAP)
        st.plotly_chart(fig_age_b, use_container_width=True)
        
with tab4:
    st.header("Voting Patterns vs Round Table Composition")
    st.write(
        "Compare how votes are distributed across gender and race "
        "relative to the cast composition of a given episode for the selected season(s)."
    )

    selected_round = st.selectbox("Select Round Table", options=sorted(votes_df['round_table'].unique()))

    round_votes = votes_df[
        (votes_df['round_table'] == selected_round) &
        (votes_df['Season'].astype(str).isin([str(s) for s in selected_seasons]))
    ]

    if round_votes.empty:
        st.warning("No votes for this round/season.")
        st.stop()

    f_votes = round_votes[round_votes['Season'].astype(str).isin([str(s) for s in selected_seasons])]

    # Filter for the selected round
    round_data = f_votes[f_votes['round_table'] == selected_round]

    room_gender = round_data[['player', 'voter_gender']].drop_duplicates().groupby('voter_gender').size().reset_index(name='player_count')
    room_race = round_data[['player', 'voter_ethnicity']].drop_duplicates().groupby('voter_ethnicity').size().reset_index(name='player_count')

    votes_gender = round_data.groupby('target_gender')['target'].count().reset_index(name='votes_received')
    votes_race = round_data.groupby('target_ethnicity')['target'].count().reset_index(name='votes_received')

    total_players = room_gender['player_count'].sum()  # Total unique players at the round table
    st.metric("Total Contestants Analysed", total_players)

    col1, col2 = st.columns(2)
    GENDER_COLORS = {
    "female": "#e377c2",
    "male": "#1f77b4"
    }

    RACE_COLORS = {
        "white": "#f1690e",
        "person_of_color": "#ffdf61"
    }
    # Race comparison
    with col1:
        st.subheader("Race")
        fig_race = go.Figure()
        fig_race.add_trace(go.Pie(
            labels=room_race['voter_ethnicity'],
            values=room_race['player_count'],
            hole=0.6, domain={'x': [0, 0.45]},
            title=f"Room Mix (Ep {selected_round})",
            marker=dict(colors=[RACE_COLORS[g] for g in room_race['voter_ethnicity']])
        ))
        fig_race.add_trace(go.Pie(
            labels=votes_race['target_ethnicity'],
            values=votes_race['votes_received'],
            hole=0.6, domain={'x': [0.55, 1]},
            title="Votes Received",
            marker=dict(colors=[RACE_COLORS[g] for g in votes_race['target_ethnicity']])
        ))
        st.plotly_chart(fig_race, use_container_width=True)
    
    # Gender comparison
    with col2:
        st.subheader("Gender")
        fig_gender = go.Figure()
        fig_gender.add_trace(go.Pie(
            labels=room_gender['voter_gender'],
            values=room_gender['player_count'],
            hole=0.6, domain={'x': [0, 0.45]},
            title=f"Room Mix (Ep {selected_round})",
            marker=dict(colors=[GENDER_COLORS[g] for g in room_gender['voter_gender']])
        ))
        fig_gender.add_trace(go.Pie(
            labels=votes_gender['target_gender'],
            values=votes_gender['votes_received'],
            hole=0.6, domain={'x': [0.55, 1]},
            title="Votes Received",
            marker=dict(colors=[GENDER_COLORS[g] for g in votes_gender['target_gender']])
        ))
        st.plotly_chart(fig_gender, use_container_width=True)
       
        
        
        

st.info("""
        **Disclaimer on Demographic Data**
        
        Gender and racial data was inferred using AI-based name analysis and may not reflect the self-identification of the contestants (libraries: `ethnicolr`, `gender_guesser`).
        
        For the purposes of this analysis, the term person of color is used to mean non-white, in accordance with the APA Style guidelines on bias-free language [[APA Style: Racial and Ethnic Identity](https://apastyle.apa.org/style-grammar-guidelines/bias-free-language/racial-ethnic-minorities)].
        
        Every effort was made to cross-reference AI-generated labels with self-identification data where available. This was a personally challenging exercise, and I sincerely welcome any corrections or suggestions to improve accuracy. Full datasets and code are available on [GitHub](https://github.com/xoxor/theTraitors).
""")

st.divider()
st.subheader("Analysis of the data")
t1, t2, t3, t4 = st.tabs(["Early Banishments", "Survival Time", "Age Analysis", "Voting Patterns"])

with t1:
    st.markdown("""
    This section compares the initial cast composition against those who were banished in the first four episodes. 
    
    **Race Impact:**
    Although People of Color make up less than a third of the starting cast (28.8%), they account for nearly half of early banishments (46.7%) in all UK seasons combined. This disparity suggests that players of color may be disproportionately suspected or targeted in the early stages of the game, potentially reflecting unconscious racial bias in initial trust assessments.
    
    **Gender Impact:**
    Despite a nearly even gender split in the starting cast, female players experience a slightly higher rate of early banishment (53.3%) in all UK seasons combined. This shift may indicate unconscious gender-based assumptions influencing early judgments about trustworthiness or threat perception.""")
with t2:
    st.markdown("""
    This section analyzes how long different demographic groups tend to survive in the game and their representation among finalists.
    To better isolate unconscious bias, the following analysis excludes the celebrity season. Because celebrity players enter the game with established public images and, in many cases, preexisting relationships, these factors likely reduce certain forms of unconscious bias examined here (while potentially introducing different ones). The conclusions below are therefore drawn from this premise.
    
    **Survival Longevity:** Across both genders, White contestants consistently survive longer than People of Color. Moreover, white men have the highest average survival rate (nearly 8 episodes), while women of color have the lowest survival longevity (just over 6 episodes), suggesting compounded effects of racial and gender bias over time.
    
    **Finalists Gender Composition:** Men are in average more likely to reach the finale (56.7%). This could potentially reflect gendered perceptions of leadership, threat, or credibility as the game progresses.
    
    **Finalists Racial Composition:** White contestants are disproportionately represented among finalists relative to the starting cast (81.7%). This indicates that racial biases persist throughout the game, influencing who is ultimately perceived as trustworthy or non-threatening enough to win.""")
with t3:
    st.markdown("""
    This section examines the age distribution of contestants and how it relates to their survival in the game.
    
    While older contestants represent a smaller share of the starting cast, their lower average episode survival suggests a potential age-related disadvantage. However, given the smaller sample size and possible game-mechanics confounds, this pattern should be interpreted cautiously as suggestive rather than definitive evidence of age-based bias.""")
with t4:
    st.markdown("""
    This section compares the demographic composition of the contestants at a given round table (episode) against the distribution of votes received in that episode. 
    As with the previous analysis, celebrity seasons are excluded to better isolate unconscious social biases. In non-celebrity play, contestants lack the buffer of public personas or preexisting relationships, making early voting behavior a clearer reflection of instinctive suspicion. This is particularly evident in the very first round table, where players explicitly describe voting based on “gut feeling,” as little concrete evidence has yet emerged.

    **Gender Impact:** At the first round table, the gender composition of the room is perfectly balanced: 50% men and 50% women. Despite this parity, voting behavior diverges sharply. A striking 76.8% of all votes were cast against female players, with men receiving just 23.2%. This disparity is far too large to be explained by representation alone and suggests that, in the earliest phase of the game, women are significantly more likely to be perceived as suspicious or expendable. This pattern holds across all seasons except the third, where the gap is narrower.

    **Race Impact:** By contrast, the racial composition of the room at this stage is 73.2% white and 26.8% people of color, and voting by race mirrors this baseline: white players received 69.5% of votes and players of color 30.5%. The race gap still exists but not as large as for the early banishments. In fact, because banishments are determined by plurality rather than proportional vote share, even a relatively small number of votes can be decisive when concentrated on a single individual.

    However, if we look at the second and third round tables (to extend the analysis to episode 4 and compare it with the banishment analysis) the gender gap in voting narrows and racial disparities in targeting tend to widen, suggesting a shift in how suspicion is socially distributed as the game progresses.""")