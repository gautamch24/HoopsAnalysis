import os
import asyncio
import aiofiles
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
import time


seasons = list(range(2024,2025))


data_dir = "CurrentSeasonData"
current_standings_dir = os.path.join(data_dir, "current_standings")
current_scores_dir = os.path.join(data_dir, "current_scores")


if not os.path.exists(current_standings_dir):
    os.makedirs(current_standings_dir)
if not os.path.exists(current_scores_dir):
    os.makedirs(current_scores_dir)

##helps get the html and iterate through each page to grab the html
async def get_html(url,selector, sleep =5, retries = 3):
    html = None
    for i in range(retries):
        await asyncio.sleep(sleep*i)

        try: 
            async with async_playwright() as p:
                browser = await p.firefox.launch()
                page = await browser.new_page()
                await page.goto(url)
                print(await page.title())
                html = await page.inner_html(selector)

        except PlaywrightTimeout:
                print(f"Timeout error on {url}")
                continue
        except Exception as e:
             print(f"Error fetching HTML from {url}: {e}")
        finally:
            if browser:
                 await browser.close()
        break
    return html


async def scrape_season(season):
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
    html = await get_html(url, "#content .filter")
    soup = BeautifulSoup(html, "html.parser")
    links = soup.find_all("a", href = True)

    href = [l["href"] for l in links]
    ##all the code above is used to get the html for the seasons so october 2016, november 2016, and so on. beautiful soup is used to process the links and html and find all <a> tags since they contain the "/leagues/NBA_YEAR_games-month.html"
    standings_pages = [f"https://basketball-reference.com{l}" for l in href]
    for url in standings_pages:
         save_path = os.path.join(current_standings_dir, url.split("/")[-1])
         if os.path.exists(save_path):
              continue
         html = await get_html(url, "#all_schedule")
         async with aiofiles.open(save_path,"w") as f:
            await f.write(html)


async def main1():
     tasks = [scrape_season(season) for season in seasons]
     await asyncio.gather(*tasks)

async def scrape_game(standings_file):
    async with aiofiles.open(standings_file,'r') as f:
        html = await f.read()
    soup = BeautifulSoup(html,"html.parser")
    links = soup.find_all("a")
    hrefs = [l.get("href") for l in links]
    ##this line is what grabs the specfic html links to each boxscore page that have boxscore and .html in the line
    box_scores = [l for l in hrefs if l and "boxscore" in l and ".html" in l]
    ##this concatenates the full og link with the box-score tag we got in the line above to give us a full path to that box score page
    box_scores = [f"https://www.basketball-reference.com{l}" for l in box_scores]

    for url in box_scores:
        save_path = os.path.join(current_scores_dir,url.split("/")[-1])
        if os.path.exists(save_path):
            continue

        html = await get_html(url, "#content")

        if not html:
            continue

        async with aiofiles.open(save_path ,"w") as f:
            await f.write(html)

async def main2():
    tasks = [scrape_season(season) for season in seasons]
    await asyncio.gather(*tasks)
    standings_files = os.listdir(current_standings_dir)
    for file in standings_files:
        await scrape_game(os.path.join(current_standings_dir, file))

asyncio.run(main2())