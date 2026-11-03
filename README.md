# Nightlife Autopilot — Group 7

Prepared app skeleton; NOT deployed. No Railway screenshot or app/repository URL has been created.

## Included

`schema.sql`: PostgreSQL CREATE statements for venues, app_users, and scout_reports. The last two tables are empty future-feature placeholders.

Browser search form → Python `/api/venues` → parameterized PostgreSQL query → counts and venue cards. Backend calculations count matching venues, recorded opening hours, and confirmed outdoor seating. Counts use all matches; cards display up to 100. `/health` checks the database and venues table.

## Source data

1,511 named records exported from `osm_venues` in your uploaded ZIP, with original OSM IDs. Unnamed records excluded. These are collected records, not the synthetic CSVs. The exact 10-club sample linked in your Collected Data PDF is not attached: verify/add that sample before submitting if it differs from the included records. The source uses a bounding box, so records can extend beyond Manhattan. Listings may be outdated. Retain OpenStreetMap attribution under ODbL.

## Local practice

Python 3.10+. From this folder:

```bash
export DATABASE_MODE=sqlite
python3 seed.py
python3 app.py
```

Visit http://localhost:8000. This is a local SQLite demonstration; it does not fulfill Railway deployment. For PostgreSQL, install requirements.txt and set DATABASE_URL privately instead.

## Railway deployment checklist

Generic steps; follow your class Brightspace walkthrough where it differs. The login-protected walkthrough was not accessed. Dashboard labels may vary.

1. Upload this folder's CONTENTS to the root of your team GitHub repository. If keeping the ML project, put this app in a subfolder and set Railway's app root accordingly. Never commit .env files or credentials.
2. Open your team's Railway project. Add PostgreSQL if you do not already have the class database.
3. Add an app service from the team GitHub repository. The included Dockerfile installs the driver and starts with `python seed.py && exec python app.py`.
4. Set DATABASE_URL on the APP service to a Railway reference to the PostgreSQL service's connection variable, using private networking. Keep this value private. Do not set DATABASE_MODE on Railway.
5. Deploy. The seed applies schema.sql and imports data. Repeated imports preserve existing OSM IDs and never delete other rows. If existing tables named venues, app_users, or scout_reports have incompatible columns, resolve that conflict before deployment; CREATE IF NOT EXISTS does not migrate old schemas.
6. Generate a public domain for the APP service. Open the app and /health from another browser. Search a venue and change the ZIP filter to confirm counts respond.
7. In the Railway PostgreSQL table view, verify all three tables. A fresh database has 1,511 venue rows; existing databases may have more because seed does not delete records.
8. Capture a genuine Railway table screenshot, cropping out database URLs, credentials, and personal information. Do not use a local preview as the Railway screenshot.

No database connection string is exposed by the app. User input is demonstrated through database search and computed counts; no user accounts or crowd predictions are implemented yet.

## What to submit

Database assignment:
- Actual Railway table screenshot (still needed).
- schema.sql, the exact schema used by this app.
- Never submit the database public connection URL.

App assignment:
- Public APP URL, with frontend and backend connected to Railway PostgreSQL (still needed).
- Actual team GitHub repository link (still needed).

No presentation required.

## Validation

Run `python3 -m unittest discover -s tests -v` from this directory. Local tests cover repeat imports, filters/counts, literal wildcard and SQL injection inputs, HTTP routes, and error responses that hide connection details. Tests use SQLite; a live PostgreSQL deployment still needs verification. This is a classroom skeleton, not a production service.
