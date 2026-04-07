Genre Classifier that predicts the tzanetakis genre using data from the [AcousticBrainz Database](https://data.metabrainz.org/pub/musicbrainz/acousticbrainz/dumps/acousticbrainz-highlevel-json-20220623/).

Data can be found [here](https://uofc-my.sharepoint.com/:f:/g/personal/nolan_wick_ucalgary_ca/IgDMvN6U3muQTYvTF6CeXjhgAdRdoW8nxhPL0UyKJdDvceY).

Download the whole Docker if needed and decompress the pg_data.7z file inside, otherwise just download the acousticbrainz_dump folder and apply it to a local PostgreSQL (16) database.

The data exploration can be run with EDA.py
The Random Forest can be ran with RF.py.
The MLP can be ran with MLP.py
