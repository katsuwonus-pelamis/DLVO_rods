#classes are supposed to work at fixed concentration ?
import numpy as np
from dataclasses import dataclass, field
from functools import lru_cache
from abc import ABC, abstractmethod

def circle_overlap_area(r1: float, r2: float, d: float) -> float:
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return float(np.pi * min(r1, r2) ** 2)

    term1 = r1**2 * np.arccos((d**2 + r1**2 - r2**2) / (2 * d * r1))
    term2 = r2**2 * np.arccos((d**2 + r2**2 - r1**2) / (2 * d * r2))
    term3 = 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    return term1 + term2 - term3

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
    molecular_volume: float
    charge_fraction: float
    delta: float
    
    @lru_cache(maxsize=128)
    def ionic_strength(self, concentration: float) -> float:
        return self.cmc + self.charge_fraction * (concentration - self.cmc)
    
    @lru_cache(maxsize=128)
    def D_CTAC(self)-> float:
        return 2*(3*self.aggregation_number*self.molecular_volume/(4*np.pi))**(1/3) 
    
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
        return self.D_CTAC() + 2.0 * self.delta / kappa

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
        for surf in self.surfactants:
            total += surf.ionic_strength(concentration)
        return total


@dataclass(frozen=True)
class RodSpecies:
    name: str
    width: float
    length: float
    hamaker: float
    


### ------------------------
### ACTUAL POTENTIALS
### ------------------------

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


@dataclass(frozen=True)
class VdWInteraction(InteractionModel):
   
    def pair_energy_per_length(self, a, b, separation, solution, concentration):
        r1 = a.width / 2.0
        r2 = b.width / 2.0
        gap = separation - r1 - r2
        if gap <= 0:
            return np.inf

        hamaker = np.sqrt(a.hamaker*b.hamaker)
        return -hamaker / (12.0 * np.sqrt(2.0) * gap**1.5) * np.sqrt(r1 * r2 / (r1 + r2))

@dataclass(frozen=True)
class ElectrostaticInteraction(InteractionModel):
    def pair_energy_per_length(self, a, b, separation, solution, concentration):
        r1 = a.width / 2.0 + solution.layer_thickness
        r2 = b.width / 2.0 + solution.layer_thickness
        gap = separation - r1 - r2
        if gap <= 0:
            return np.inf

        ionic_strength = solution.total_ionic_strength(concentration)
        kappa = solution.inverse_debye(ionic_strength)

        return np.sqrt(kappa / (2.0 * np.pi) * (r1 * r2 / (r1 + r2))) * self.Z(concentration, solution) * np.exp(-kappa * gap)
    
    def Z(self, concentration: float, solution) -> float:
        return 64.0 * np.pi*solution.constants.eps_0 * solution.eps_r * (solution.constants.k_B * solution.temperature / solution.constants.e_charge)** 2 *np.tanh(solution.constants.e_charge *solution.zeta_pot* solution.total_ionic_strength(concentration)/ (4.0 * solution.constants.k_B * solution.temperature)) ** 2
    
    
    
@dataclass(frozen=True)
class DepletionInteraction(InteractionModel):
    depletant: Surfactant

    def pair_energy_per_length(self, a, b, separation, solution, concentration):
        depl_D = self.depletant.effective_depletant_diameter(concentration, solution)
        t_eff = solution.layer_thickness   #we need to sort this out, more specifically need to look for the actual delta - keep in mind that in principle it's a different surfactant from the one forming mycels

        r1 = a.width / 2.0 + t_eff + depl_D / 2.0
        r2 = b.width / 2.0 + t_eff + depl_D / 2.0

        if separation >= r1 + r2:
            return 0.0

        overlap = circle_overlap_area(r1, r2, separation)
        return -self.depletant.osmotic_pressure(concentration, solution) * overlap
    
    
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
# Build your current system
# ---------------------------------------------------------------------

def build_default_model():
    constants = PhysicalConstants()

    solution = SolutionState(
        temperature=293.0,
        eps_r=80.10,
        ctac_thickness=3.2e-9,
        constants=constants,
        surfactants=[],
    )

    ctac = CTACModel(
        name="CTAC",
        cmc=1.31,
        aggregation_number=120.0,
        molecular_volume=4.309e-28,
        charge_fraction=0.28,
        depletant_core_diameter=4.62e-9,
        delta=0.725,
    )

    second_surfactant = CTACModel(
        name="SurfactantB",
        cmc=0.8,
        aggregation_number=90.0,
        molecular_volume=3.8e-28,
        charge_fraction=0.12,
        depletant_core_diameter=5.5e-9,
        delta=0.6,
        ionic_strength_prefactor=0.15,
    )

    solution.surfactants = [ctac, second_surfactant]

    big = RodSpecies(
        name="big",
        width=44e-9,
        length=131e-9,
        hamaker=40e-20,
        surface=SurfaceState(thickness=3.2e-9, potential=0.035),
    )

    small = RodSpecies(
        name="small",
        width=13e-9,
        length=131e-9,
        hamaker=40e-20,
        surface=SurfaceState(thickness=3.2e-9, potential=0.035),
    )

    species = {"big": big, "small": small}

    interactions = [
        VdWInteraction(),
        ElectrostaticInteraction(),
        DepletionInteraction([ctac, second_surfactant]),
    ]

    hex_big = PhaseRecipe(
        name="hex_big",
        pair_terms=[PairTerm(count=6, species_a="big", species_b="big", separation_scale=1.0, length_mode="a")],
        double_counting_factor=2.0,
        particle_normalization=1.0,
    )

    hex_small = PhaseRecipe(
        name="hex_small",
        pair_terms=[PairTerm(count=6, species_a="small", species_b="small", separation_scale=1.0, length_mode="a")],
        double_counting_factor=2.0,
        particle_normalization=1.0,
    )

    s1 = PhaseRecipe(
        name="s1",
        pair_terms=[
            PairTerm(count=4, species_a="big", species_b="big", separation_scale=1.0, length_mode="a"),
            PairTerm(count=8, species_a="big", species_b="small", separation_scale=np.sqrt(2.0), length_mode="min"),
        ],
        double_counting_factor=2.0,
        particle_normalization=2.0,
    )

    sigma = PhaseRecipe(
        name="sigma",
        pair_terms=[
            PairTerm(count=40, species_a="big", species_b="big", separation_scale=1.0, length_mode="a"),
            PairTerm(count=32, species_a="big", species_b="small", separation_scale=np.sqrt(2.0), length_mode="min"),
        ],
        double_counting_factor=2.0,
        particle_normalization=12.0,
    )

    return solution, species, interactions, {
        "hex_big": hex_big,
        "hex_small": hex_small,
        "s1": s1,
        "sigma": sigma,
    }


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
