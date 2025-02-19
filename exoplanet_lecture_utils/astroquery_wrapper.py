import pandas as pd
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

def get_exoplanet_data(local=True):
    """
    This function queries the NASA Exoplanet Archive for exoplanets data where
     the planet's mass in Jupiter masses (pl_bmassj) and radius in Earth radii
     (pl_rade) are both greater than 0.
    The data is then converted to a pandas DataFrame and returned.

    Args:
        local (bool): Whether to use the local database or not.

    Returns:
        pnasa (pd.DataFrame): A pandas DataFrame containing the queried
        exoplanet data.
    """

    if local:
        pnasa = pd.read_csv('https://share.phys.ethz.ch/~ipa/'
                            'exoplanet_lecture_FS24/PSCompPars_2024.csv')
    else:
        # Info with what we can query
        # https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html
        # Query all the exoplanet info
        # nasa_ea = NasaExoplanetArchive.query_criteria('pscomppars',where="(pl_bmassj > 0) AND (pl_rade > 0) ");
        nasa_ea = NasaExoplanetArchive.query_criteria('pscomppars')
        # Let us store the data as a pandas table
        pnasa = nasa_ea.to_pandas()

    return pnasa