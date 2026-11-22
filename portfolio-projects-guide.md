# Portfolio Projects: Implementation Guide

Step by step plan for the 9 portfolio projects. Each one has what you're building, the stack, the steps, and when it counts as done.

## Rules for every project

- One GitHub repo per project with a README: the problem, a screenshot or gif, how to run it, how it works, results.
- Build the ugly working version first, then improve it. Get one thing working end to end before adding features.
- Deploy every project, even a free tier. A live link beats a repo with no demo.
- Free hosting that works fine: Render, Railway, or Fly.io for the backend, Vercel for the frontend, Supabase for Postgres.
- Commit small and often so the history shows how you built it.
- Write a few tests for the core logic (the forecast function, the wallet math, the stock calculation). Not everything, just what would hurt if it broke.

## 1. Fuel price tracker with prediction

**What:** Track Nigerian fuel prices (petrol and diesel) over time and forecast where they're going.

**Stack:** Python, pandas, statsmodels or Prophet, FastAPI, Postgres, and a simple chart frontend (Streamlit is fastest, React if you want it polished).

**Steps:**
1. Find your data. Monthly average retail price reports from the National Bureau of Statistics are a solid source, plus price listing sites and news. Start with whatever history you can get, even if it's only a few years.
2. Clean it into one table: date, product, price, source. Fix gaps and obvious typos.
3. Store it in Postgres. If you want it connected to StationAPI, reuse the fuel price tables there.
4. Build a baseline first: "next month = this month" and a moving average. Every fancy model has to beat these.
5. Try one real model: ARIMA or Prophet. Then try gradient boosting with lag features (last 1, 2, 3 months) plus outside features like exchange rate and crude price.
6. Validate properly with walk-forward testing: train on the past, predict the next step, move forward, repeat. Never shuffle time series data randomly.
7. Add a FastAPI endpoint like `/forecast?product=petrol&months=3` that returns predictions with a range, not just one number.
8. Build the dashboard: history line, forecast line, and the error metric shown honestly.
9. Add a scheduled job that pulls new data monthly.

**Done when:** the model beats the baseline on MAE in walk-forward testing, it's deployed, and the README shows the comparison.

## 2. Naira FX rate forecaster

**What:** Track official vs parallel market naira rates and forecast the trend.

**Stack:** Same as the fuel tracker. Build them with shared code since the pipeline is nearly identical.

**Steps:**
1. Collect data: official rates from the CBN and parallel rates from a public tracker or dataset. Check each site's terms before scraping. Daily data is best.
2. Build a table with date, official rate, parallel rate, and compute the spread (the gap) as its own column.
3. Plot everything first. Look at the jumps, the policy change dates, the trends.
4. Baseline: tomorrow equals today (random walk). This is very hard to beat for FX, so don't be surprised if your model barely does.
5. Forecast direction and ranges instead of exact numbers. Predicting "will the spread widen over the next 2 weeks" is more honest and more useful.
6. Try ARIMA or Prophet, then lag features with gradient boosting. Mark big policy events as features.
7. Use walk-forward validation and report both error and directional accuracy.
8. Expose it through the same API pattern as the fuel tracker, and chart it.
9. Put a clear note on the page that it's a forecast experiment, not financial advice.

**Done when:** you have a live chart, a forecast with uncertainty bands, and an honest comparison against the random walk baseline. An honest "this barely beats baseline" write-up is still a good portfolio piece.

**Tip:** feed the FX series into the fuel tracker as a feature. Fuel prices respond to the exchange rate, and the two projects then link together.

## 3. RAG chatbot over Nigerian documents

**What:** A chatbot that answers questions from a document set and shows where each answer came from.

**Stack:** Python, PyMuPDF for PDF text, sentence-transformers or an embedding API, Chroma or pgvector, an LLM API, FastAPI, a simple chat page.

**Steps:**
1. Pick the document set. Tax laws, the JAMB syllabus, or a university handbook. Pick one with a clear audience. Use official current PDFs, since old versions give wrong answers.
2. Extract the text with PyMuPDF and keep the section headings and page numbers.
3. Chunk it: around 400 to 600 tokens with a bit of overlap, split at section boundaries when possible. Store the source, section, and page with each chunk.
4. Embed the chunks and store them in Chroma (easiest) or pgvector (if you want it all in Postgres).
5. Build retrieval: embed the question, get the top 4 to 6 chunks.
6. Prompt the LLM with the chunks and tell it to answer only from them, cite the section, and say "I can't find that in the documents" when it can't.
7. Make a test set of about 30 real questions with known answers. Run it after every change so you can see if you actually improved anything.
8. Improve what fails: better chunk size, add reranking, add keyword search alongside the vector search.
9. Wrap it in a FastAPI endpoint and a basic chat page that shows the source passages under each answer.

**Done when:** answers cite their source, it refuses when the answer isn't in the documents, and your README shows the test set score.

## 4. Self-playing game bot with RL

**What:** An agent that learns to play a game by trial and error.

**Stack:** Python, Gymnasium, Stable-Baselines3, PyTorch, TensorBoard.

**Steps:**
1. Learn the loop: observe, act, get reward, repeat. Read the Gymnasium basics.
2. Train PPO on CartPole just to see the whole pipeline work. It takes minutes.
3. Pick your real game. A simple one is best: Snake, Flappy Bird, or a small grid game you write yourself as a Gymnasium environment.
4. Define the observation (what the bot sees), the actions, and the reward. This is the hardest and most important part. Start with simple rewards, then shape them if it gets stuck.
5. Train with PPO or DQN and watch the reward curve in TensorBoard.
6. If it doesn't learn: check the reward first, then the observation, then the hyperparameters, in that order.
7. Record a video of the trained agent playing, and a clip from early training for contrast.
8. Write up the training curve, what you tried, what failed, and why.

**Done when:** you have a video of the bot playing well, the training curve, and the code to reproduce it.

Do this one before the drift car. The same skills and setup carry straight over.

## 5. AI car that learns to drift

**What:** A simulated car that learns to hold a controlled slide around a track or circle.

**Stack:** Python, pygame or a simple renderer, Gymnasium, Stable-Baselines3. (Unity ML-Agents is the heavier option if you want 3D later.)

**Steps:**
1. Start in 2D top-down. Don't touch 3D until the 2D version works.
2. Build a simple car physics model: position, heading, velocity, steering, throttle. Add tire grip with a slip angle so the car can actually slide when pushed.
3. Wrap it as a Gymnasium environment. Observations: speed, heading vs travel direction (the slip angle), distance to the track center, track direction ahead. Actions: steering and throttle.
4. First goal is easy: just drive around a circle without crashing. Train PPO until that works.
5. Now the drift reward: reward speed while the slip angle stays in a target range (for example 20 to 50 degrees) and the car stays near the track line. Punish crashes and spinouts.
6. Train in stages. Easy grip first, then lower grip. Circle first, then a figure eight, then a track.
7. Add a debug view that draws the slip angle and the reward so you can see what the agent is optimizing.
8. Watch for reward hacking. If the car just spins in place, your reward is wrong, so fix the reward.
9. Record a video of a clean drift lap.

**Done when:** the car holds a drift around a corner or circle and you have the clip to show it.

## 6. Fuel availability and queue map

**What:** A map where people report whether a station has fuel and how long the queue is. It builds on StationAPI.

**Stack:** Your existing FastAPI backend, Postgres, Leaflet with OpenStreetMap on the frontend, built as a PWA.

**Steps:**
1. Seed the stations. Pull fuel stations for your area from OpenStreetMap (the Overpass API, tag `amenity=fuel`) and load them into StationAPI.
2. Add a `reports` table: station id, user id, has fuel (yes or no), queue level (none, short, long), created at.
3. Add `POST /reports` (needs login, reuse your JWT) and `GET /stations/status` which returns each station's current state.
4. The status logic: only count reports from the last 2 to 3 hours, weigh newer ones more, and show "unknown" when there's no recent report. Never show stale data as current.
5. Build the map page: pins colored by status, tap a pin to see details and a big two tap report button.
6. Design for bad internet: keep payloads tiny, cache the station list, queue reports offline and send them when the connection returns (service worker plus IndexedDB).
7. Fight spam: rate limit per user, and require two or more agreeing reports before a status flips if you see abuse.
8. Test it with real people. Get 10 friends on campus or in town to use it for a week and see what breaks.

**Done when:** real reports flow in, the map updates, and it still works when the connection drops.

## 7. SME stock and sales tracker

**What:** A simple tool for small shops to log sales, track stock, and see low stock alerts.

**Stack:** FastAPI, Postgres, and a mobile friendly PWA frontend (React or plain HTML and JS).

**Steps:**
1. Before you code, talk to 2 or 3 real shop owners. Ask how they track stock now and what annoys them. Build for that, not for what you imagine.
2. Design the tables: `products`, `stock_movements` (every stock in or out as a row), `sales`, `sale_items`. Don't store a single stock number that you overwrite. Compute stock from the movements so there's always a history and you can fix mistakes.
3. Build the core: add a product, record stock received, record a sale (which creates the stock out movements).
4. Add the daily summary: total sales, profit if cost price is entered, best sellers, items running low.
5. Add a low stock threshold per product and show a clear alert list on the home screen.
6. Make recording a sale take as few taps as possible. That speed is the real feature.
7. Make it work offline and sync later, since shop connections are unreliable. Show naira amounts clearly.
8. Later ideas: record sales from a text message or WhatsApp, and a weekly report.

**Done when:** one real shop uses it for a week and you've fixed what they complained about.

## 8. Power outage tracker

**What:** People report when the light goes off or comes back, and the app shows live status and typical patterns per area.

**Stack:** FastAPI, Postgres, a simple map or area list frontend, pandas for the pattern analysis.

**Steps:**
1. Keep reporting to one tap: "light is on" or "light is off", tied to an area the user picks once.
2. Tables: `areas`, `reports` (area, status, timestamp, user), and an hourly `area_status` summary you compute.
3. Compute the current status per area from recent reports, and show how long it's been in that state.
4. Build the pattern view: for each area, chance of having light by hour of the day and day of the week. This is just an average over past data at first.
5. This project needs data to be interesting. Seed it with friends in a few areas (campus, hostels, nearby streets) and let it run for a few weeks before you build any model.
6. Once you have data: forecast "chance of light in the next few hours" using the hour and weekday averages as the baseline, then try a simple model with extra features.
7. Add alerts later (notify me when the light returns in my area).
8. Handle bad or fake reports with rate limits and by trusting agreement between users.

**Done when:** several areas have real data, the pattern chart shows something real, and the baseline forecast is running.

## 9. Coin flip gambling site (points only)

**What:** Users get free points daily, bet them on a coin flip, and keep the winnings in their balance.

**Stack:** FastAPI and Postgres (you already know both), with a plain frontend or React. The FARM stack works too if you want the practice.

**Steps:**
1. Tables: `users`, `wallets` (balance), `transactions` (every change as a row: type, amount, balance after), `flips`.
2. Auth with JWT, like StationAPI.
3. Daily claim: store `last_claim_at`, allow one claim per 24 hours using server time only, never the client clock.
4. The flip endpoint: check the bet is positive, within your min and max, and not more than the balance. Do the whole thing inside one database transaction with a row lock on the wallet so two fast requests can't spend the same points twice.
5. Use Python's `secrets` module for the coin flip, not `random`. Add an idempotency key so a double tap doesn't flip twice.
6. Record every outcome in `transactions`. The balance should always equal the sum of the transactions, and you can write a test for that.
7. Make it fair in a way users can check: commit to a hashed server seed before the flip, combine it with a client seed and a nonce, and reveal the seed afterward (the provably fair pattern).
8. Frontend: balance, bet input, a flip animation, history, daily claim button, and a leaderboard of top balances.
9. Add rate limits and basic abuse checks (many accounts claiming daily points).

**Keep it legal:** points should not be redeemable for money, and people shouldn't be able to buy them with money. The moment real money is involved, gambling regulation applies. If you ever go that way, check the rules first.

**Done when:** the wallet never goes negative or out of sync under rapid requests, and the fairness check works.

## Suggested order

1. **Coin flip site.** Fastest win (about a week) and good practice with transactions and wallets.
2. **Fuel availability and queue map.** Builds on StationAPI, so you start ahead.
3. **SME stock and sales tracker.** Reuses what you learned in the wallet and ledger work.
4. **Fuel price tracker, then the FX forecaster.** Share one data pipeline, and use FX as a fuel feature.
5. **RAG chatbot.** Self contained and very relevant right now.
6. **Self-playing game bot, then the drift car.** Do them in that order since the drift car reuses everything.
7. **Power outage tracker.** Start collecting data early in the background, since it needs weeks of reports before it's interesting.

Two projects at a time is plenty. Finish and deploy one before starting the next.
