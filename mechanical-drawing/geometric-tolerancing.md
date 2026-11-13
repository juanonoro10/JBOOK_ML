---
title: "Geometric Tolerancing"
---

**What it is.** Two parts dimensioned with geometric dimensioning and tolerancing (GD&T). Drawing P-01 is a small commercial bronze pin (UNS C22000), with a flange and an end cut at 35°, drawn at scale 6:1. Drawing H-02 is a steel flanged hub (SAE 3310), with a bolt circle of eight holes, an M42×4.5 thread and a central bore. Dimensions are in millimeters.

**What it is for.** Size tolerances alone do not control how features are located or oriented with respect to each other. In a pin like P-01, the flange must be coaxial with the shank. In a hub like H-02, the bore must turn true with respect to the mounting face, and the spigot must be square to it so the hub seats and centers correctly. GD&T states these functional requirements explicitly and makes them inspectable.

**What the drawing shows.**

- **Datum features.** Each part has a primary datum (A) and a secondary datum (B), chosen from the features that locate the part in its assembly. The controls are referenced to them.
- **P-01:** a **concentricity control** of the flange diameter relative to the shank axis (datum A), a **profile control** of the inclined end face (view A) referenced to A and B and defined by a basic dimension, and a surface finish requirement of 0.2 µm on all surfaces.
- **H-02:** **circular runout** of the Ø20 bore relative to A and B, **perpendicularity** of the Ø35 spigot at maximum material condition (Ⓜ), which permits extra tolerance as the feature departs from its MMC size, a 0.03 mm **flatness** control on the flange face, and a basic bolt circle diameter (Ø88) for the eight holes.

```{figure} img/geometric-tolerancing-1.png
:label: fig-gdt-pin

Sheet 1 (P-01): bronze pin with concentricity and profile controls.
```

```{figure} img/geometric-tolerancing-2.png
:label: fig-gdt-hub

Sheet 2 (H-02): flanged hub with runout, perpendicularity at MMC and a basic bolt circle.
```

{download}`Download the drawings (PDF) <pdf/geometric-tolerancing.pdf>`
