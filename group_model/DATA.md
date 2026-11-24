# Data: Yelp Open Dataset, open bars

## Dataset description

| | |
|---|---|
| Source | Yelp Open Dataset, business file `yelp_academic_dataset_business.json` (https://www.yelp.com/dataset). The script downloads a public mirror of the same file from Hugging Face. |
| Real or synthetic | Real. Yelp listings and crowd-sourced attributes, not scraped by us and not synthetic. |
| Terms | Yelp's data agreement allows academic, non-commercial use as part of a course. It forbids redistributing the data, so the file is not committed; the script downloads it. |
| Period | Snapshot released by Yelp in January 2022; review counts are cumulative to that date. |
| Coverage | 11 North American metro areas: Philadelphia, Tampa, New Orleans, Saint Louis, Nashville, Indianapolis, Tucson, Reno, Edmonton, Boise, Santa Barbara (and their suburbs). Not New York. |
| One row per | Open bar |
| Filter | 150,346 businesses → categories include both "Bars" and "Nightlife" and `is_open = 1` (7,528) → price level and noise level recorded (5,406) |
| Rows used | 5,406 |
| Split | random 70/30, seed 42: 3,784 train, 1,622 test |

**Why the price/noise filter.** On Yelp, busy bars get more of their attributes filled in by reviewers.
If "noise level unknown" were a feature, it would mostly measure popularity itself, which is
leakage. Keeping only bars with both fields recorded removes that.

**Why city is not a feature.** The product launches in New York, which is not in this dataset. With
no city terms, the model predicts from venue traits alone, so it can score a New York bar.

## Data dictionary

All features are 0/1 flags built by `features_from_raw()` in the script. "Share" is the share of the
5,406 bars with the flag set to 1.

| Column | Role | Meaning | Yelp source | Share |
|---|---|---|---|---|
| `review_count` | **Outcome** | Number of Yelp reviews (popularity). Mean 196.9, median 109, 75th percentile 234, max 4,876. Modeled as log(1 + reviews). | `review_count` | |
| `business_id`, `name`, `city`, `state` | Identifier, not used | | same | |
| `stars` | Not used | Average star rating; left out because it is a review outcome, not a trait the owner controls | `stars` | |
| `price_cheap` | Feature | Price level $ | `RestaurantsPriceRange2` = 1 | 19.3% |
| `price_high` | Feature | Price level $$$ or $$$$ | `RestaurantsPriceRange2` = 3 or 4 | 6.9% |
| (price $$) | Reference level | in the intercept | `RestaurantsPriceRange2` = 2 | 73.8% |
| `noise_quiet` | Feature | Crowd-reported noise: quiet | `NoiseLevel` | 4.9% |
| `noise_loud` | Feature | Crowd-reported noise: loud | `NoiseLevel` | 17.3% |
| `noise_very_loud` | Feature | Crowd-reported noise: very loud | `NoiseLevel` | 4.9% |
| (noise average) | Reference level | in the intercept | `NoiseLevel` | 72.9% |
| `music_dj` | Feature | Has a DJ | `Music.dj` | 3.2% |
| `music_live` | Feature | Live music | `Music.live` | 21.2% |
| `music_karaoke` | Feature | Karaoke | `Music.karaoke` | 0.8% |
| `music_jukebox` | Feature | Jukebox | `Music.jukebox` | 7.2% |
| `good_for_dancing` | Feature | Good for dancing | `GoodForDancing` | 8.7% |
| `happy_hour` | Feature | Has happy hour | `HappyHour` | 62.1% |
| `closes_late_weekend` | Feature | Closes at or after 1 am on Friday or Saturday | `hours` | 35.6% |
| `good_for_groups` | Feature | Good for groups | `RestaurantsGoodForGroups` | 95.0% |
| `outdoor_seating` | Feature | Outdoor seating | `OutdoorSeating` | 64.2% |
| `has_tv` | Feature | Has TVs | `HasTV` | 88.5% |
| `serves_food` | Feature | Also listed as a restaurant | `categories` contains "Restaurants" | 78.5% |
| `cocktail_bar` | Feature | Bar type | `categories` "Cocktail Bars" | 16.0% |
| `dive_bar` | Feature | Bar type | "Dive Bars" | 7.2% |
| `sports_bar` | Feature | Bar type | "Sports Bars" | 22.0% |
| `wine_bar` | Feature | Bar type | "Wine Bars" | 10.6% |
| `pub` | Feature | Bar type | "Pubs" | 17.5% |
| `lounge` | Feature | Bar type | "Lounges" | 10.2% |
| `dance_club` | Feature | Bar type | "Dance Clubs" | 2.6% |
| `beer_bar` | Feature | Bar type | "Beer Bar" | 8.2% |
| `music_venue` | Feature | Bar type | "Music Venues" | 6.0% |

A missing Yelp attribute (for example no `Music` field) counts as 0 for that flag.

## What this data does not have

Yelp has no live data: no crowd level tonight, no line length, no cover charge. Those are what
Scouts report in the app. This model fills the gap before Scouts exist, by predicting which bars
groups will care about most.
