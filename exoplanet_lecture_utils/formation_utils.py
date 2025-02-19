from astropy.io.votable import parse
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
from copy import deepcopy
from scipy.stats import median_abs_deviation

def votable_to_pandas(votable_file):
    votable = parse(votable_file)
    table = votable.get_first_table().to_table(use_names_over_ids=True)
    table['_sed_flux'] = table['_sed_flux']*table['_sed_freq']
    table['_sed_eflux'] = table['_sed_eflux']*table['_sed_freq']
    table['_sed_wav'] = (table['_sed_freq']).to(u.micron, equivalencies=u.spectral()).value
    return table.to_pandas()


def clean_data(sigma, name, plot, data_dict):
    mean_pos = np.array((np.median(data_dict[name]['_RAJ2000']), np.median(data_dict[name]['_DEJ2000'])))
    std_pos = np.array((median_abs_deviation(data_dict[name]['_RAJ2000']),
                        median_abs_deviation(data_dict[name]['_DEJ2000'])))

    keep = np.sqrt(((data_dict[name]['_RAJ2000'] - mean_pos[0]) ** 2 / std_pos[0] ** 2 + (
                data_dict[name]['_DEJ2000'] - mean_pos[1]) ** 2 / std_pos[1] ** 2)) <= sigma

    if plot:
        fig, ax = plt.subplots(ncols=2, figsize=(8, 4))

        ax[0].scatter(data_dict[name]['_RAJ2000'][keep], data_dict[name]['_DEJ2000'][keep], color='k')
        ax[0].scatter(data_dict[name]['_RAJ2000'][np.invert(keep)], data_dict[name]['_DEJ2000'][np.invert(keep)],
                      color='tab:orange')
        ax[0].scatter(mean_pos[0], mean_pos[1], color='r')

        ax[1].scatter(np.log10(data_dict[name]['_sed_freq'][~keep]),
                      np.log10(data_dict[name]['_sed_flux'][~keep]), color='tab:orange', alpha=0.5)
        ax[1].scatter(np.log10(data_dict[name]['_sed_freq'][keep]),
                      np.log10(data_dict[name]['_sed_flux'][keep]), color='k')

        plt.xlabel('log10(Frequency) [Hz]')
        plt.ylabel(r'log10(Flux) [$\mathrm{W/m^2/Hz}$]')

        plt.tight_layout()

        plt.show()

    result = deepcopy(data_dict)
    result[name] = result[name][keep]

    return result

def load_spectra(names, path):
    data_dict = {n: votable_to_pandas(path + n + '_vizier_votable.vot')
                 for n in names}

    data_dict = clean_data(data_dict=data_dict,
                           name='doar_24',
                           sigma=1,
                           plot=False)
    
    data_dict = clean_data(data_dict=data_dict,
                           name='sr_4',
                           sigma=1,
                           plot=False)
    
    data_dict = clean_data(data_dict=data_dict,
                           name='sr_9',
                           sigma=1.495,
                           plot=False)
    
    data_dict = clean_data(data_dict=data_dict,
                           name='sr_3',
                           sigma=5.,
                           plot=False)
    
    data_dict = clean_data(data_dict=data_dict,
                           name='HD_164595',
                           sigma=2.,
                           plot=False)
    
    data_dict = clean_data(data_dict=data_dict,
                           name='HD_186302',
                           sigma=1.,
                           plot=False)
    return data_dict