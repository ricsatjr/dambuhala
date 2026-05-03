# dambuhala

**Dambuhala** is a Filipino adjective describing something of extreme size — an apt description of large dams and the scale of their potential impacts.

`dambuhala` is an open-source Python toolkit for the rapid preliminary assessment of hydrological hazards associated with large dams. It is designed for data-scarce conditions, working with freely available global datasets without requiring proprietary model inputs or licensed software. Outputs are compatible with standard open-source GIS applications, particularly QGIS.

---

## Scope

`dambuhala` currently addresses two hazard pathways:

1. **Dam breach flood hydrograph** — peak discharge and time-varying outflow estimated from multiple published empirical models, with explicit uncertainty bounds
2. **Downstream flood inundation** — spatial extent and water depth from breach hydrograph routing using LISFLOOD-FP

The tool is developed in the context of Philippine large dams but is applicable to similar settings elsewhere in Southeast Asia.

---

## Quick Start

```bash
git clone https://github.com/ricsatjr/dambuhala.git
cd dambuhala
pip install numpy scipy matplotlib
```

*Full installation instructions and usage examples are in the [Wiki](../../wiki).*

---

## Documentation

Detailed technical documentation is maintained in the [project Wiki](../../wiki), including:

- Design principles and architecture
- Development roadmap
- Empirical model descriptions and references
- Data sources and acquisition
- LISFLOOD-FP setup
- Usage examples

---

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE)

## Disclaimer

This software and its documentation are provided for preliminary assessment purposes only. Portions of the documentation were drafted with AI assistance and are pending full technical review. See [DISCLAIMER.md](DISCLAIMER.md) for full details.
