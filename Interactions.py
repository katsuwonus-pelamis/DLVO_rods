import numpy as np
from dataclasses import dataclass, field
from functools import lru_cache
from abc import ABC, abstractmethod

def circle_overlap_area(r1: float, r2: float, d: float) -> float:
    d = np.asarray(d, dtype=float)
    # ensure r1,r2 are scalars
    r1 = float(np.asarray(r1))
    r2 = float(np.asarray(r2))

    sumr = r1 + r2
    diffr = abs(r1 - r2)

    area = np.zeros_like(d, dtype=float)

    mask_no_overlap = d >= sumr
    mask_full_containment = d <= diffr
    mask_partial = ~(mask_no_overlap | mask_full_containment)

    if np.any(mask_full_containment):
        area[mask_full_containment] = np.pi * min(r1, r2) ** 2

    if np.any(mask_partial):
        dp = d[mask_partial]
        # safe arccos arguments clipped to [-1, 1]
        arg1 = (dp**2 + r1**2 - r2**2) / (2 * dp * r1)
        arg2 = (dp**2 + r2**2 - r1**2) / (2 * dp * r2)
        arg1 = np.clip(arg1, -1.0, 1.0)
        arg2 = np.clip(arg2, -1.0, 1.0)

        term1 = r1**2 * np.arccos(arg1)
        term2 = r2**2 * np.arccos(arg2)
        term3 = 0.5 * np.sqrt(
            (-dp + r1 + r2) * (dp + r1 - r2) * (dp - r1 + r2) * (dp + r1 + r2)
        )
        area[mask_partial] = term1 + term2 - term3
    #print(area)
    return area.item() if area.ndim == 0 else area

@dataclass(frozen=True)
class PhysicalConstants:
    eps_0: float = 8.8541878188e-12
    e_charge: float = 1.602176634e-19
    k_B: float = 1.380649e-23
    N_A: float = 6.02214076e23

@dataclass(frozen=True)
class Surfactant:
    cmc: float
    aggregation_number: float
    mycel_diameter: float
    charge_fraction: float
    delta: float
    
    @lru_cache(maxsize=128)
    def ionic_strength(self, concentration: float) -> float:
        return self.cmc + self.charge_fraction * (concentration - self.cmc)
    
    #arguments to be made on whether to hard code it as a member since we kind of have it....
    #@lru_cache(maxsize=128)
    #def D_mycel(self)-> float:
    #    #return 2*(3*self.aggregation_number*self.molecular_volume/(4*np.pi))**(1/3) 
    #    print(np.cbrt(6*self.molecular_volume))
     #   return np.cbrt(6*self.molecular_volume)
    
    @lru_cache(maxsize=128)
    def n_micelles(self, concentration: float, constants: PhysicalConstants) -> float:
        if concentration < self.cmc:
            return 0.0
        return constants.N_A / self.aggregation_number * (concentration - self.cmc)

    #we'll have to sort this out since now it's not the same depletant as in the mycels
    #@lru_cache(maxsize=128)
    #def effective_thickness(self, concentration: float, env: "SolutionState") -> float:
        kappa = env.inverse_debye(self.ionic_strength(concentration))
     #   return env.layer_thickness + self.delta / kappa

    @lru_cache(maxsize=128)
    def effective_depletant_diameter(self, concentration: float, env: "SolutionState") -> float:
        kappa = env.inverse_debye(self.ionic_strength(concentration))
        return self.mycel_diameter + 2.0 * self.delta / kappa

    @lru_cache(maxsize=128)
    def phi_eff(self, concentration: float, env: "SolutionState") -> float:
        n = self.n_micelles(concentration, env.constants)
        D_eff = self.effective_depletant_diameter(concentration, env)
        return n * (4.0 / 3.0) * np.pi * (D_eff / 2.0) ** 3

    @lru_cache(maxsize=128)
    def osmotic_pressure(self, concentration: float, env: "SolutionState") -> float:
        n = self.n_micelles(concentration, env.constants)
        phi = self.phi_eff(concentration, env)
        return n * env.constants.k_B * env.temperature * (1 + phi + phi**2 - phi**3) * (1 - phi) ** (-3)


@dataclass(frozen=True)
class SolutionState:
    temperature: float
    eps_r: float
    layer_thickness: float
    zeta_pot: float
    composition: float
    constants: PhysicalConstants = field(default_factory=PhysicalConstants)
    surfactants: tuple[Surfactant, ...] = field(default_factory=tuple)
    

    @lru_cache(maxsize=128)
    def inverse_debye(self, concentration: float) -> float:
        return np.sqrt(
            concentration
            * self.constants.N_A
            * self.constants.e_charge**2
            / (self.constants.eps_0 * self.eps_r * self.constants.k_B * self.temperature)
        )
        
    @lru_cache(maxsize=128)
    def total_ionic_strength(self, concentration: float) -> float:
        total = 0 
        total += self.surfactants[0].ionic_strength(self.composition*concentration)
        total += self.surfactants[1].ionic_strength((1 - self.composition)*concentration)
        return total


@dataclass(frozen=True)
class RodSpecies:
    width: float
    length: float
    hamaker: float
    


### ------------------------
### ACTUAL POTENTIALS
### ------------------------
'''
class InteractionModel(ABC):
    @abstractmethod
    def pair_energy_per_length(
        self,
        a: RodSpecies,
        b: RodSpecies,
        separation: float,
        solution: SolutionState,
        concentration: float,
    ) -> float:
        raise NotImplementedError
'''

@dataclass(frozen=True)
class VdWInteraction:
     def energy(self, a, b, separation):
        separation = np.asarray(separation, dtype=float)

        r1 = a.width / 2.0
        r2 = b.width / 2.0
        gap = separation - r1 - r2

        hamaker = np.sqrt(a.hamaker * b.hamaker)
        prefactor = -hamaker / (12.0 * np.sqrt(2.0)) * np.sqrt(r1 * r2 / (r1 + r2))

        energy = np.full_like(gap, np.inf, dtype=float)
        mask = gap > 0
        energy[mask] = prefactor / gap[mask] ** 1.5

        return energy.item() if energy.ndim == 0 else energy

@dataclass(frozen=True)
class ElectrostaticInteraction:
    def energy(self, a, b, separation, solution, concentration):
        separation = np.asarray(separation, dtype=float)

        r1 = a.width / 2.0 + solution.layer_thickness
        r2 = b.width / 2.0 + solution.layer_thickness
        gap = separation - r1 - r2
        ionic_strength = solution.total_ionic_strength(concentration)
        kappa = solution.inverse_debye(ionic_strength)
        prefactor = np.sqrt(kappa / (2.0 * np.pi) * (r1 * r2 / (r1 + r2))) * self.Z(solution)

        energy = np.full_like(gap, np.inf, dtype=float)
        mask = gap > 0
        energy[mask] = prefactor * np.exp(-kappa * gap[mask])

        return energy.item() if energy.ndim == 0 else energy
    
    def Z(self, solution) -> float:
        return 64.0 * np.pi*solution.constants.eps_0 * solution.eps_r * (solution.constants.k_B * solution.temperature / solution.constants.e_charge)** 2 *np.tanh(solution.constants.e_charge *solution.zeta_pot/ (4.0 * solution.constants.k_B * solution.temperature)) ** 2
    
    
    
@dataclass(frozen=True)
class DepletionInteraction:
    def energy(self, a, b, separation, solution, concentration):
        sep = np.asarray(separation, dtype=float)
        t_eff = solution.layer_thickness

        energy = np.zeros_like(sep, dtype=float)

        # two surfactants: fractions = composition and (1 - composition)
        fracs = (solution.composition, 1.0 - solution.composition)
        for idx, frac in enumerate(fracs):
            surf = solution.surfactants[idx]
            conc_i = frac * concentration
            depl_D = surf.effective_depletant_diameter(conc_i, solution)

            r1 = a.width / 2.0 + t_eff + depl_D / 2.0
            r2 = b.width / 2.0 + t_eff + depl_D / 2.0
            overlap = circle_overlap_area(r1, r2, sep)
            pressure = surf.osmotic_pressure(conc_i, solution)
            print(pressure)
            energy += -pressure * overlap

        return energy.item() if energy.ndim == 0 else energy
    
    
'''
# ---------------------------------------------------------------------
# Phase recipes
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class PairTerm:
    count: int
    species_a: str
    species_b: str
    separation_scale: float = 1.0
    length_mode: str = "min"


@dataclass(frozen=True)
class PhaseRecipe:
    name: str
    pair_terms: Sequence[PairTerm]
    double_counting_factor: float = 2.0
    particle_normalization: float = 1.0

    def length_for_pair(self, a: RodSpecies, b: RodSpecies, mode: str) -> float:
        if mode == "a":
            return a.length
        if mode == "b":
            return b.length
        return min(a.length, b.length)


@dataclass
class PhaseModel:
    species: dict[str, RodSpecies]
    interactions: Sequence[InteractionModel]
    recipe: PhaseRecipe

    def energy(self, separation: float, solution: SolutionState, concentration: float) -> float:
        total = 0.0
        for term in self.recipe.pair_terms:
            a = self.species[term.species_a]
            b = self.species[term.species_b]
            d_eff = separation / term.separation_scale
            L = self.recipe.length_for_pair(a, b, term.length_mode)

            pair_energy = 0.0
            for interaction in self.interactions:
                pair_energy += interaction.pair_energy_per_length(a, b, d_eff, solution, concentration)

            total += term.count * pair_energy * L

        total /= self.recipe.double_counting_factor
        total /= self.recipe.particle_normalization
        return total




# ---------------------------------------------------------------------
# Scanning helpers
# ---------------------------------------------------------------------

def scan_phase_minimum(model: PhaseModel, solution: SolutionState, concentrations: np.ndarray, lattice: np.ndarray):
    minima = []
    for c in concentrations:
        energies = np.array([model.energy(d, solution, c) for d in lattice]) / (solution.constants.k_B * solution.temperature)
        idx = int(np.argmin(energies))
        minima.append([lattice[idx], energies[idx], c])
    return np.asarray(minima)
'''
