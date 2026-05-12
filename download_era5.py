import cdsapi
import os

c = cdsapi.Client()

os.makedirs('data/raw/era5', exist_ok=True)

# Télécharger mois par mois
mois = ['01','02','03','04','05','06',
        '07','08','09','10','11','12']

for mois_num in mois:
    output_file = f'data/raw/era5/era5_med_2022_{mois_num}.nc'
    
    # Skip si déjà téléchargé
    if os.path.exists(output_file):
        print(f"  → Mois {mois_num} déjà téléchargé, on passe.")
        continue
    
    print(f"\nTéléchargement mois {mois_num}/12 ...")
    
    c.retrieve(
        'derived-era5-land-daily-statistics',
        {
            'variable': [
                '2m_temperature',
                '2m_dewpoint_temperature',
                '10m_u_component_of_wind',
                '10m_v_component_of_wind',
            ],
            'year': '2022',
            'month': [mois_num],
            'day': [str(d).zfill(2) for d in range(1, 32)],
            'daily_statistic': 'daily_mean',
            'time_zone': 'UTC+00:00',
            'frequency': '1_hourly',
            'area': [47, -5, 35, 28],
            'format': 'netcdf',
        },
        output_file
    )
    print(f"  → Mois {mois_num} téléchargé ✅")

print("\nTous les mois téléchargés !")