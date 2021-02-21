from pathlib import Path
import os

class Config(object):
    model_params = {
        "shiedling_reduction_between_groups": 0.25,
        "shielding_increase_within_group": 2,
        "default_quarantine_period": 5,
        "better_hygiene_infection_scale": 0.7,
    }
    CONTACT_MATRIX_DIR = Path(os.path.dirname(__file__)) / "contact_matrices"
    compartment_index = {'S': 0,
             'E': 1,
             'I': 2,
             'A': 3,
             'R': 4,
             'H': 5,
             'C': 6,
             'D': 7,
             'O': 8,
             'Q': 9,
             'U': 10,
             'CS': 11,
             'CE': 12,
             'CI': 13,
             'CA': 14,
             'CR': 15,
             'CH': 16,
             'CC': 17,
             'CD': 18,
             'CO': 19,
             'CQ': 20,
             'CU': 21,
             'Ninf': 22,
             }
    longname = {'S': 'Susceptible',
                'E': 'Exposed',
                'I': 'Infected (symptomatic)',
                'A': 'Asymptomatically Infected',
                'R': 'Recovered',
                'H': 'Hospitalised',
                'C': 'Critical',
                'D': 'Deaths',
                'O': 'Offsite',
                'Q': 'Quarantined',
                'U': 'No ICU Care',
                # 'CS': 'Change in Susceptible',
                # 'CE': 'Change in Exposed',
                # 'CI': 'Change in Infected (symptomatic)',
                # 'CA': 'Change in Asymptomatically Infected',
                # 'CR': 'Change in Recovered',
                # 'CH': 'Change in Hospitalised',
                # 'CC': 'Change in Critical',
                # 'CD': 'Change in Deaths',
                # 'CO': 'Change in Offsite',
                # 'CQ': 'Change in Quarantined',
                # 'CU': 'Change in No ICU Care',
                # 'Ninf': 'Change in total active infections',  # sum of E, I, A
                }
    shortname = {'S': 'Sus.',
                 'E': 'Exp.',
                 'I': 'Inf. (symp.)',
                 'A': 'Asym.',
                 'R': 'Rec.',
                 'H': 'Hosp.',
                 'C': 'Crit.',
                 'D': 'Deaths',
                 'O': 'Offsite',
                 'Q': 'Quar.',
                 'U': 'No ICU',
                 'CS': 'Change in Sus.',
                 'CE': 'Change in Exp.',
                 'CI': 'Change in Inf. (symp.)',
                 'CA': 'Change in Asym.',
                 'CR': 'Change in Rec.',
                 'CH': 'Change in Hosp.',
                 'CC': 'Change in Crit.',
                 'CD': 'Change in Deaths',
                 'CO': 'Change in Offsite',
                 'CQ': 'Change in Quar.',
                 'CU': 'Change in No ICU',
                 'Ninf': 'New Infected',  # newly exposed to the disease = - change in susceptibles
                 }
