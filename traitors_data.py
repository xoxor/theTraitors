import pandas as pd
import requests
import gender_guesser.detector as gender
from ethnicolr import pred_wiki_name
from io import StringIO
import re
from bs4 import BeautifulSoup
from scipy import io

def get_traitors_automated_data(url, season):
    # Scrape the right table from the Wikipedia page
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    if response.status_code != 200: 
        print(f"Failed to retrieve data from {url}")
        return None

    html = response.text.replace("“", '"').replace("”", '"')
    html = re.sub(r'(rowspan|colspan)="(\d+)[^"]*"', r'\1="\2"', html)
    tables = pd.read_html(StringIO(html))
    df = next((t for t in tables if 'Age' in t.columns), None)
    if df is None: return None
    
    # Remove Wiki markers and split into First/Last for ethnicolr
    
    df['Contestant'] = df['Contestant'].replace(r'\[.*\]', '', regex=True).str.strip()
    
    df['fname'] = df['Contestant'].apply(lambda x: x.split()[0])
    df['lname'] = df['Contestant'].apply(lambda x: ' '.join(x.split()[1:]) if len(x.split()) > 1 else '')
    df['player_id'] = f"{season}_" + df['Contestant']
    status_col = next((c for c in df.columns if c.startswith('Finish')), None)

    if status_col:
        pattern = r'(?P<Stat>.*?) \((?:Episode )?(?P<Ep>\d+)\)'
        extracted = df[status_col].str.extract(pattern)
        
        df['Finish'] = extracted['Stat'].fillna(df[status_col])
        df['Episode'] = pd.to_numeric(extracted['Ep'], errors='coerce')

        if status_col != 'Finish':
            df = df.drop(columns=[status_col])
    else:
        print("No 'Finish' column found for Episode extraction.")
    
    
    # Gender Inference
    d = gender.Detector()
    df['Inferred_Gender'] = df['fname'].apply(d.get_gender)

    # Ethnicity Inference
    try:
        cols_before = df.columns.tolist()

        df = pred_wiki_name(df, 'lname', 'fname')
        
        cols_after = df.columns.tolist()
        new_cols = [c for c in cols_after if c not in cols_before]

        df = df.rename(columns={'race': 'Inferred_Ethnicity'})
        cols_to_drop = [c for c in new_cols if c != 'race']
        df = df.drop(columns=cols_to_drop, errors='ignore')

    except Exception as e:
        print(f"Ethnicity inference failed: {e}")
        df['Inferred_Ethnicity'] = "Unknown"

    
    return df


def get_votes(url, season):
   
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    soup = BeautifulSoup(resp.text, 'html.parser')
    tables = soup.find_all('table', class_='wikitable')
    all_data = []

    for table in tables:
        rows = table.find_all("tr")
        
        # Attempt to get episode numbers from the first row
        header_row = rows[0]
        episodes = []
        for th in header_row.find_all("th"):
            colspan = int(th.get("colspan", 1))
            text = th.get_text(strip=True)
            # Only keep numeric episode headers
            if text.isdigit():
                episodes.extend([text]*colspan)

        # Skip tables with no episode info
        if not episodes:
            continue

        # Process each player row
        for row in rows[1:]:
            cells = row.find_all(['td', 'th'])
            if len(cells) < 2:
                continue
            
            player = None
            for cell in cells:
                if cell.name == "th" and cell.get_text(strip=True) not in ["Traitors'Decision", "Immune", "Banishment", "Vote"]:
                    player = cell.get_text(strip=True)
                    break
            if not player:
                continue

            ep_index = 0
            for cell in cells:
                if cell.name == "td":
                    text = cell.get_text(strip=True).replace("\n", ", ")
                    target_id = f"{season}_{text}"
                    colspan = int(cell.get("colspan", 1))
                    for _ in range(colspan):
                        if ep_index < len(episodes):
                            id = f"{season}_{player}"
                            all_data.append({
                                "player": id,
                                "target": target_id,
                                "round_table": episodes[ep_index]
                            })
                            ep_index += 1

    df = pd.DataFrame(all_data)
    return df


    
def get_data_per_season(url, season_number, celebrity):
    df = get_traitors_automated_data(url, season_number)
    ds = get_votes(url, season_number)
    if df is not None and not celebrity:
        df['Season'] = season_number
        ds['Season'] = season_number
    elif df is not None and celebrity:
        df['Season'] = f"C{season_number}"
        ds['Season'] = f"C{season_number}"
    return df, ds


def get_all_seasons_data(base_url, country, num_seasons, celebrity=False):
    for season in range(1, num_seasons+1):  
        season_url = f"{base_url}{season}" if not celebrity else base_url
        season_df, season_ds = get_data_per_season(season_url, season, celebrity)
        if season_df is not None:
            name = f"{country}_traitors_season_{season}_ai_tagged.csv"
            season_df.to_csv(name, index=False)
            name2 = f"{country}_traitors_season_{season}_votes.csv"
            season_ds.to_csv(name2, index=False)
            
    
if __name__ == "__main__":
    uk_base_url = "https://en.wikipedia.org/wiki/The_Traitors_(British_TV_series)_series_"
    get_all_seasons_data(uk_base_url, "UK", 4)
    celebrity_url = "https://en.wikipedia.org/wiki/The_Celebrity_Traitors"
    get_all_seasons_data(celebrity_url, "UK_Celebrity", 1, celebrity=True)


