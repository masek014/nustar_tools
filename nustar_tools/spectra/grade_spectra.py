import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from astropy.coordinates import SkyCoord
from astropy.time import Time
from dataclasses import dataclass

from nustar_tools.mapping import maps
from nustar_tools.utils import utilities
from nustar_tools.pixels.PixelArray import PixelArray

from . import pha_manipulation as pm


PILEUP_GRADE_CORRECTIONS = {
    '0'   : lambda a, b : a - 0.25 * b,
    '1-4' : lambda a, b : a - b,
    '0-4' : lambda a, b : a - 1.25 * b
}


def make_grade_distribution(
    evt_file: str,
    time_interval: tuple[Time, Time | str, str] = None,
    region_kwargs: dict = None
) -> dict[str, int]:
    """
    Count the number of events for each grade. A temporal and/or spatial filter
    can be applied.
    """
    
    evt_data, hdr = utilities.get_event_data(evt_file, time_interval)

    if region_kwargs is not None:
        nustar_map = maps.make_nustar_map(evt_data, hdr, time_interval)
        region = region_kwargs['Class'](
            SkyCoord(*region_kwargs['center'], frame=nustar_map.coordinate_frame),
            **region_kwargs['kwargs']
        )
        data = PixelArray(evt_data, hdr, map_=nustar_map, region=region).evts
    
    else:
        data = evt_data

    grades, counts = np.unique(data['GRADE'].data, return_counts=True)
    grades, counts = list(grades), list(counts)
    
    # Fill in missing grades.
    for grade in range(0, 32):
        if grade not in grades:
            grades.insert(grade, grade)
            counts.insert(grade, 0)
    
    grade_dict = {str(g): int(c) for g, c in zip(grades, counts)}

    return grade_dict


@dataclass
class GradeSpectrum():
    grade_file: str
    grade: str
    fpm: str
    kwargs: dict


    @property
    def spectrum(self) -> tuple[u.Quantity, u.Quantity]:
        values, edges, _ = pm.read_pha_spectrum(self.grade_file)
        return values, edges

    @property
    def label(self):
        return f'FPM {self.fpm}, Grade {self.grade}'

    
    def plot(self, ax: plt.Axes = None) -> plt.Axes:

        counts, edges = self.spectrum
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6), layout='constrained')
        ax.stairs(
            counts.value,
            edges.value,
            label=self.label,
            **self.kwargs
        )
        ax.set(
            xlabel=f'Energy [{edges.unit}]',
            ylabel=f'Spectrum [{counts.unit}]'
        )

        return ax


@dataclass
class GradeCollection():
    grade_path_format: str
    grades: list[str]
    fpms: list[str]


    def prepare_data(
        self,
        fpmA_kwargs: dict = {},
        fpmB_kwargs: dict = {}
    ):
        
        default_fpmA_kwargs = dict(
            cmap='turbo'
        )
        default_fpmB_kwargs = dict(
            cmap='plasma'
        )

        fpmA_kwargs = {**default_fpmA_kwargs, **fpmA_kwargs}
        fpmB_kwargs = {**default_fpmB_kwargs, **fpmB_kwargs}

        kwargs = dict(
            A = fpmA_kwargs,
            B = fpmB_kwargs
        )

        self.data = {}
        total_spectra = len(self.fpms) * len(self.grades)
        count = 0
        for fpm in self.fpms:
            self.data[fpm] = {}
            cmap = matplotlib.colormaps[kwargs[fpm].pop('cmap')]
            for grade in self.grades:
                grade_path = self.grade_path_format.format(fpm=fpm, grade=grade)
                color = cmap(count/total_spectra)
                spectrum = GradeSpectrum(
                    grade_path,
                    grade,
                    fpm,
                    dict(color=color, **kwargs[fpm])
                )
                self.data[fpm][grade] = spectrum
                count += 1


    def plot_spectra(
        self,
        ax: plt.Axes = None,
        plot_grades: list[str] = 'all',
        pileup_correction: bool = False,
        pileup_kwargs: dict = {}
    ) -> plt.Axes:
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6), layout='constrained')

        if plot_grades == 'all':
            plot_grades = self.grades

        for fpm in self.fpms:
            for grade in plot_grades:
                spectrum = self.data[fpm][grade]
                spectrum.plot(ax=ax)

                if pileup_correction:
                    default_pileup_kwargs = dict(ls='--')
                    if '21-24' in self.data[fpm]:
                        if grade in PILEUP_GRADE_CORRECTIONS:
                            expression = PILEUP_GRADE_CORRECTIONS[grade]
                            counts, edges = spectrum.spectrum
                            pileup_counts = self.data[fpm]['21-24'].spectrum[0]
                            corrected = expression(counts, pileup_counts)
  
                            kwargs = {**spectrum.kwargs, **default_pileup_kwargs}
                            kwargs = {**kwargs, **pileup_kwargs}

                            ax.stairs(
                                corrected.value,
                                edges.value,
                                label=f'{spectrum.label} pileup-corrected',
                                **kwargs
                            )
                    else:
                        print('Pileup correction was specified, but grades '
                              '21-24 were not provided in initalization. '
                              'Not performing correction.')
        
        return ax
    

    def plot_ratios(
        self,
        ax: plt.Axes = None,
        reference_grade: str = '0'
    ) -> plt.Axes:
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,3), layout='constrained')

        for fpm in self.fpms:
            ref_spectrum = self.data[fpm][reference_grade]
            ref_counts, ref_edges = ref_spectrum.spectrum

            for grade in self.grades:
                if grade == reference_grade:
                    continue
                spectrum = self.data[fpm][grade]
                counts, edges = spectrum.spectrum
                if not np.array_equal(ref_edges, edges):
                    raise ValueError(f'The reference grade edges are different '
                                    'from the {grade} grade edges.')    
                
                ratio = counts / ref_counts
                ax.stairs(
                    ratio.value,
                    edges.value,
                    label=spectrum.label,
                    **spectrum.kwargs
                )
        
        ax.set(
            xlabel=f'Energy [{edges.unit}]',
            ylabel='Ratio',
            title=f'Referenced to grade {reference_grade}'
        )

        return ax