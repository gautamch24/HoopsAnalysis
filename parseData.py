import os 
import io
import pandas as pd
from bs4 import BeautifulSoup
import asyncio
import aiofiles
score_dir = "Scraping/Data/scores"

box_scores = os.listdir(score_dir)

box_scores = [os.path.join(score_dir,f ) for f in box_scores if f.endswith("html")]


async def parse_html(box_score):
    try:
        async with aiofiles.open(box_score) as f:
            html = await f.read()
        soup = BeautifulSoup(html, "html.parser")
        [s.decompose() for s in soup.select("tr.over_header")]
        [s.decompose() for s in soup.select("tr.thead")]
        return soup
    except Exception as e:
        print(f"Error parsing HTML in {box_score}: {e}")
        return None

async def read_line_score(soup):
    try:
        line_score_html = str(soup.find("table", attrs={"id": "line_score"}))
        line_score_list = pd.read_html(io.StringIO(line_score_html))
        if line_score_list:  
            line_score = line_score_list[0]
            if isinstance(line_score, pd.DataFrame):  
                cols = list(line_score.columns)
                cols[0] = "team"
                cols[-1] = "total"
                line_score.columns = cols
                line_score = line_score[["team", "total"]]
                return line_score
            else:
                print("No line score table found.")
        else:
            print("No line score table found.")
    except Exception as e:
        print(f"Error reading line score table: {e}")

async def read_four_factors(soup):
    try:
        four_factor_html = str(soup.find("table", attrs = {"id": "four_factors"}))
        four_factor_list = pd.read_html(io.StringIO(four_factor_html))
        if four_factor_list:
            four_factor = four_factor_list[0]
            if isinstance(four_factor, pd.DataFrame):
                return four_factor
            else:
                print("No Four Factor table found")

        else:
            print("No Four Factor table found")
    except Exception as e:
        print(f"Error reading four factor table: {e}")

async def read_season(soup):
    nav = soup.select("#bottom_nav_container")[0]
    hrefs = [a["href"]for a in nav.find_all("a")]
    season = os.path.basename(hrefs[1]).split("_")[0]
    return season




async def organize_data():
    games = []
    for box_score in box_scores:
        soup = await parse_html(box_score)
        if soup:
            line_scores  = await read_line_score(soup)
            four_factors = await read_four_factors(soup)
            home_game = pd.concat([line_scores,four_factors], axis =1, join = 'inner')
            home_game = home_game.drop(columns = ['Unnamed: 0'])
            home_game["Home"] = [0,1]
            opp_game = home_game.iloc[::-1].reset_index()
            opp_game.columns += "_opp"
            full_game = pd.concat([home_game, opp_game], axis=1)
            full_game["season"] = await read_season(soup)
            full_game["date"] = os.path.basename(box_score)[:8]
            full_game["date"] = pd.to_datetime(full_game["date"], format = "%Y%m%d")
            full_game["won"] = full_game["total"] > full_game["total_opp"]
            games.append(full_game)

            if len(games) %100 ==0:
                print(f"{len(games)}/ {len(box_scores)}")


            totalGames_DF = pd.concat(games,ignore_index=True)
            totalGames_DF.to_csv("pastNbaGames.csv")


asyncio.run(organize_data())