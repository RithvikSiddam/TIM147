import pandas as pd

# === CONFIGURATION ===
INPUT_CSV = '/Users/rithviks/Desktop/TIM147/FPA_FOD_Plus.csv'
OUTPUT_CSV = '/Users/rithviks/Desktop/TIM147/FPA_FOD_Cleaned.csv'
COLUMNS_TO_KEEP = ["DISCOVERY_DOY","FIRE_SIZE","LATITUDE","LONGITUDE","OWNER_DESCR","STATE","Des_Tp","EVT","rpms","pr_Normal","tmmn_Normal","tmmx_Normal","sph_Normal","srad_Normal","fm100_Normal","fm1000_Normal","bi_Normal","vpd_Normal","erc_Normal","DSF_PFS","EBF_PFS","EALR_PFS","EBLR_PFS","EPLR_PFS","PM25F_PFS","MHVF_PFS","LPF_PFS","NPL","RMP_PFS","TSDF_PFS","FRG","TRI_1km","Aspect_1km","Elevation_1km","Slope_1km","GHM","TPI_1km","RPL_THEMES","SDI","Annual_etr","Annual_precipitation","Annual_tempreture","Aridity_index","rmin","rmax","vs","NDVI-1day","CheatGrass","ExoticAnnualGrass","Medusahead","PoaSecunda"]
CHUNK_SIZE = 100_000  # Tweak for performance

def keep_selected_columns():
    first_chunk = True

    with open(OUTPUT_CSV, "w") as out_file:
        for chunk in pd.read_csv(INPUT_CSV, chunksize=CHUNK_SIZE):
            # Select only desired columns
            filtered_chunk = chunk[COLUMNS_TO_KEEP]

            # Write to output CSV
            filtered_chunk.to_csv(out_file, mode="a", index=False, header=first_chunk)
            first_chunk = False

    print(f"Done! Kept columns saved in: {OUTPUT_CSV}")

if __name__ == "__main__":
    keep_selected_columns()
