# download_2023.py

import cdsapi
import os

os.makedirs('data/raw/era5_2023', exist_ok=True)
os.makedirs('data/raw/firms_2023', exist_ok=True)

# --- FIRMS 2023 : téléchargement direct (gratuit, pas de compte) ---
import requests

pays = {
    'France': 'FRA',
    'Spain':  'SPA', 
    'Greece': 'GRC',
    'Morocco': 'MAR'
}

for nom, code in pays.items():
    url = (f"https://firms.modaps.eosdis.nasa.gov/"
           f"data/country/modis/2023/{code}_2023.csv")
    print(f"Téléchargement FIRMS 2023 {nom}...")
    r = requests.get(url)
    if r.status_code == 200:
        with open(f'data/raw/firms_2023/modis_2023_{nom}.csv', 'wb') as f:
            f.write(r.content)
        print(f"  ✅ {nom} OK")
    else:
        print(f"  ❌ {nom} échoué - télécharge manuellement sur firms.modaps.eosdis.nasa.gov")

# --- ERA5 2023 : mois par mois ---
c = cdsapi.Client()

for mois_num in ['01','02','03','04','05','06',
                 '07','08','09','10','11','12']:
    output = f'data/raw/era5_2023/era5_med_2023_{mois_num}.nc'
    if os.path.exists(output):
        print(f"Mois {mois_num} déjà téléchargé.")
        continue
    print(f"ERA5 2023 mois {mois_num}...")
    c.retrieve(
        'derived-era5-land-daily-statistics',
        {
            'variable': [
                '2m_temperature',
                '2m_dewpoint_temperature',
                '10m_u_component_of_wind',
                '10m_v_component_of_wind',
            ],
            'year': '2023',
            'month': [mois_num],
            'day': [str(d).zfill(2) for d in range(1, 32)],
            'daily_statistic': 'daily_mean',
            'time_zone': 'UTC+00:00',
            'frequency': '1_hourly',
            'area': [47, -5, 35, 28],
            'format': 'netcdf',
        },
        output
    )
    print(f"  ✅ Mois {mois_num} OK")

print("\nTout est prêt pour le test 2023 !")