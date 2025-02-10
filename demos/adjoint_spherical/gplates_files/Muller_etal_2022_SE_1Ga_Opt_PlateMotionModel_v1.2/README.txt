1 December 2023

This file describes the plate motion model in Muller et al. (2022). The Muller et al. (2022) model is very similar in terms of relative plate motions and plate topologies to the Merdith et al. (2021) model, with the updates and fixes described below. One of the key differences is that the model contains three reference frames, including 1) the original paleomangetic reference frame from Merdith et al. (2021), 2) a new optimised mantle reference frame (recomputed in v 1.1), and 3) a no-net rotation reference frame. If you are doing any climate-sensitive analysis, make sure you use the paleomagnetic reference frame. 

This directory contains Muller_etal_2022.gproj, which is a GPlates project file that will load the following:

- 1000_0_rotfile.rot - the Global Rotation Model but with a purely paleomagnetic reference frame
A folder called 'optimisation' which contains the mantle absolute reference frame rotations. By default these absolute rotations are used in the project file.
- 1000_410-*{Convergence/Divergence/Transform/Topologies}.gpml - the plate topologies for 1000 to 410 Ma
- 410-250_plate_boundaries.gpml - the plate topologies for 410 to 250 Ma
- 250-0_plate_boundaries.gpml - the plate topologies for 250 to 0 Ma
- TopologyBuildingBlocks.gpml - building blocks for some topologies between 410 and 0 Ma
- shapes_coastlines.gpml - a Global Coastline file (mainly for the past 400 Ma)
- shapes_continents.gpml - a Global Continent shapes file (for the past 1 Ga)
- shapes_cratons.gpml - a Global cratonic shapes file (for the past 1 Ga)
- shapes_static_polygons_Merdith_et_al.gpml - a Global static polygon file (for the past 1 Ga)
- 1000-410_poles.gpml - collection of palaeomagnetic data used to constrain the model between 1000 and 410 Ma (corresponds to Table 1 in the associated publication)

For the Mesozoic and Cenozoic, the plates comprising the Pacific Ocean traditionally move in a separate reference to the other ocean basins and continental motions, instead defined by hotspot motion. In order to preserve the plate motion of the Pacific relative to the continental domains where the new GAPWAP was implemented, we extracted relative plate rotations between the Pacific (plateID: 901) and Africa (plateID: 701) in 5 Ma intervals  between 250 and 83 Ma from the Young et al. (2019) model, which has been corrected for errata as discussed in Torsvik et al. (2019). This results in the same relative motion of all Pacific plates to continental plates, however it slightly alters the absolute position of the Pacific plates between 250 and 83 Ma. Studies interested in short(er) timescale (< 5 Ma) analysis or absolute plate motions should use a different model that explicitly links plate motion with the mantle (e.g. (Müller et al. 2019; Tetley et al. 2019)). To be clear, if you want to analyse the Pacific Ocean, including hotspot motion, Hawaiian-Emperor Bend kinematics etc. you should not use this model.

To load these datasets in GPlates do the following:

1.  Open GPlates
2.  Pull down the GPlates File menu and select Open Project
3.  Click to select the GPROJ file
4.  Click Open

Alternatively, drag and drop the GRPOJ file onto the globe.

You now have a global present day continents loaded in GPlates as well as the underlying rotation model and evolving plate topologies.  Play around with the GPlates buttons to make an animation, select features, draw features, etc. For more information, read the GPlates manual which can be downloaded from www.gplates.org.

*** IMPORTANT NOTES ***

Alterations to 1 Ga plate model of Merdith et al. 2021 (ESR) which have also been propagated to the Muller et al (2022) model:

Preamble

Some issues have been identified in the model where plate boundaries labelled 'transformed' had either a high convergence rate or high divergence rate. That is, even though they were labelled and conceptualised as transform boundaries (perhaps in some cases with an implied small amount of convergence or divergence motion), in the model they were clearly accommodation large amounts of crustal production or consumption. In the original intent of MER21++, the conceptualisation of plate boundaries was that plate motion moves orthogonally from a mid-ocean ridge towards a subduction zone (subduction does not have to be (sub)orthogonal). In a simple system, connecting a spreading ridge and subduction zone are a series of transform boundaries, (sub)parallel to the direction of movement:

—————————|>
||       |>
||       |>
——————   —|>
    ||    |>
    ||    |>
    ||    |>
    ——————|>

In this sense, reconstructing extinct ocean basins follows a simple procedure. Known subduction zones are mapped onto a continental motion model (e.g. as one defined from palaeomagnetic data), interpolation to connect regional subduction zones occurs. In ocean basins similar to the Atlantic or Indian oceans, mid-ocean ridges are inferred to be perpendicular to the direction of continental breakup (usually easiest when a supercontinent breaks up and clearly defined conjugate margins are preserved in the geological record). These mid-ocean ridges are typically easy to connect with subduction zones through a series of transform boundaries, much like the mid-Atlantic ridge connects through the Caribbean and Costa Rica with Pacific subduction, or through the Scotia-Sandiwch plates to Patagonia and the southern Pacific.

However, this approach does not work so well for large external ocean basins like the Pacific or Panthalassa Oceans, where there is a large ocean (>1/4 Earth surface) and a fragmentary record of subduction around the periphery. It is also compounded when the available data constraining plate motion of major cratons and continents are poor, as there are no constraints on both spreading rate and spreading direction. In these cases the approach of MER21++ (and also that of Domeier and Torsivk (2014) and Young et al. (2019)) is to ensure that at modelled mid-ocean ridges, divergence is occurring, and at known subduction zones, convergence is also occurring.

In this manner, the original MER21++ model mostly satisfied this constraint, however the model over-relied on transform boundaries to accommodate plate motion, such that many of the transform boundaries should be labelled (or considered) as 'inferred subduction zones'. Inferred, because the kinematic constraints of the constructed model (i.e. the location, orientation and spreading direction of synthetic mid-ocean ridges) require more subduction in order to accommodate a basic tenet of plate tectonic theory, that there is a balance of new crust created and destroyed over Ma timescales. In the original model, if one were to calculated total crustal production (spreading-ate • mid-ocean ridge length) and total crustal consumption (convergence rate • subduction zone length) they would not be equal.

Thus, the alterations offered in this update pertain to either changing the labels of some transform boundaries that were identified to have a high convergence or divergence rate or to re-align these transform boundaries to better fit the small-circle orientation defined by the plate motion. We selected a convergence/divergence rate threshold of 2 cm/a as problematic (i.e. boundaries with rates higher than this were investigated) and also a plate boundary length greater than 500 km. The majority of these boundaries occurred in ocean basins away from large cratons, a few more major transform boundaries that were changed are identified below. Some oceanic plates had their plate velocity changed to either ensure that the revised boundaries were accommodated properly or mid-ocean ridge configuration altered (summarised below). No continental plate motion was changed. The majority of the changes occurred in the Neoproterozoic, until around 600 Ma, though a few boundaries were also changed in the Palaeozoic.

These changes have resulted in bringing gross crustal production and consumption curves into much better (but still not perfect) agreement. Pragmatically, the subduction zones in the original (published) model are a stronger reflection of subduction that is preserved (and conservatively interpolated, as one might interpolate subduction along the Andean margin through amagmatic regions) in the geological record, while this altered model also contains inferred subduction zones, as necessitated by our mid-ocean spreading ridges. The difference therefore is either a reflection of arcs (principally oceanic) that are lost to time, or some idea of the inherent uncertainty within making full-plate models in deep-time (i.e. how much more unknown subduction do we need to account for kinematically balancing global plate motion).

Major boundary changes

Mawson sea-spreading removed (1000–940 Ma)
Boundary separating Tarim from Mawson-sea simplified to a single transform boundary, one arm of the Mirovoi MOR triple junction was re-aligned (directly intersecting subduction outboard of Siberia) (1000–900 Ma)
Realigned transform boundaries in Mawson Sea (940–900 Ma)
Subduction outboard of Indo-Antarctica has been removed from 900–870 Ma (still present 870–850 Ma). In the original model this was inferred subduction based on plate motion and some sparse evidence of reactivated faults after Ruker collided with the Eastern Ghats-Raynor (NB could still work if a new ocean plate introduced).
Transform boundary connecting Mirovoi MOR with Tarim replaced with a MOR boundary to make a proper triple junction (900-800 Ma)
Ocean plate subdbucting under India simplified to a one plate system (850–800 Ma)
Transform outboard of Kalarhai Craton replaced with subduction zone (800-750 Ma)
Ocean plate subducting under India separated into a 'Tarim plate' and 'Ocean plate under India' (800-750 Ma)
Mirovoi triple junction extended to 750 Ma
Transform changed to subduction zone outboard of WAC (750-700 Ma)
Cadomia-Avalonia subduction extended eastward past SM into the ocean (600–580 Ma)
The long transform outboard of Baltica as the Iapetus opened is now an oceanic subduction zone (probably the most significant change) (585–520 Ma)
Tranform outboard of western Laurentian margin now a subduction zone (500-450 Ma)
A duplicate subduction zone between 760 and 751 has been removed

Ocean plate velocity changes

Mirovoi ocean plate subjecting under Siberia has velocity increased slightly (900-850 Ma)
Mawson sea velocity changed to ensure convergence under Australia (1000-850 Ma)

Re-optimisation

The changes to plate topologies and associated relative rotations necessitated a re-optimisation of the absolute mantle reference frame, using the same parameters as described in Muller et al. (2022). The result is only insignificantly different from the original absolute plate rotations.

Other alterations

Some bugs were removed in the coastline file. 


#######
References

Domeier, M., 2016. A plate tectonic scenario for the Iapetus and Rheic oceans. Gondwana Research, 36, pp.275-295, doi: 10.1016/j.gr.2015.08.003

Domeier, M., 2018. Early Paleozoic tectonics of Asia: towards a full-plate model. Geoscience Frontiers, 9(3), pp.789-862, doi: 10.1016/j.gsf.2017.11.012

Matthews, K.J., Maloney, K.T., Zahirovic, S., Williams, S.E., Seton, M. and Mueller, R.D., 2016. Global plate boundary evolution and kinematics since the late Paleozoic. Global and Planetary Change, 146, pp.226-250, doi: 10.1016/j.gloplacha.2016.10.002

Merdith, A.S., Collins, A.S., Williams, S.E., Pisarevsky, S., Foden, J.D., Archibald, D.B., Blades, M.L., Alessio, B.L., Armistead, S., Plavsa, D. and Clark, C., 2017. A full-plate global reconstruction of the Neoproterozoic. Gondwana Research, 50, pp.84-134, doi: 10.1016/j.gr.2017.04.001

Müller, R. D., Cannon, J., Tetley, M., Williams, S. E., Cao, X., Flament, N., Bodur, Ö. F., Zahirovic, S., and Merdith, A., 2022, A tectonic-rules based mantle reference frame since 1 billion years ago–implications for supercontinent cycles and plate-mantle system evolution: Solid Earth, p. 1-42, doi: 10.5194/se-13-1127-2022

Tetley, M.G., 2018. Constraining Earth’s plate tectonic evolution through data mining and knowledge discovery. PhD Thesis

Torsvik, T.H., Steinberger, B., Shephard, G.E., Doubrovine, P.V., Gaina, C., Domeier, M., Conrad, C.P. and Sager, W.W., 2019. Pacific‐Panthalassic reconstructions: Overview, errata and the way forward. Geochemistry, Geophysics, Geosystems, 20(7), pp.3659-3689, doi: 10.1029/2019GC008402

Young, A., Flament, N., Maloney, K., Williams, S., Matthews, K., Zahirovic, S. and Müller, R.D., 2019. Global kinematics of tectonic plates and subduction zones since the late Paleozoic Era. Geoscience Frontiers, 10(3), pp.989-1013, doi: 10.1016/j.gsf.2018.05.011

Any questions, please email:

            Dietmar Muller dietmar.muller@sydney.edu.au
