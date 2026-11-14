# What differs between the datasets

- **Annotation convention splits the datasets in two.** Ours and PANDA-Karolinska outline tumour regions (a 40-um closing adds 1.4% / 2.4%, ~1 region per 256-um window, median region 0.22 / 1.1 mm^2); SICAPv2 and PANDA-Radboud outline glands (closing adds 10% / 89%, outlines 3-12x denser, median regions 0.042 / 0.006 mm^2).
- **Region-level is not the same as precise.** Karolinska's masks are drawn at low resolution and put 5.3% of the tumour label on glass (ours 0.2%, SICAPv2 0.7%, Radboud 0.2%).
- **Tumour content differs with the convention and the case mix.** Every one of our slides holds tumour (median 37% of tissue); 63-80% of the public units do, and the gland-level Radboud masks give the lowest share (15%).
- **Our stain is the most purple and least separable.** Haematoxylin and eosin directions 14 deg apart (public 20-25), tissue least red (a* 18 vs 23-37); Karolinska is the reddest and most saturated. In UNI feature space every dataset forms its own cluster (5-fold accuracy 0.999 for telling the datasets apart). Frechet distance to ours is 4.2x our own noise floor for Radboud, 4.8x for SICAPv2 and 5.2x for Karolinska, but our biopsies and our large sections are already 4.1x apart: the specimen-type gap inside our site is as large as the gap to the nearest public set.
- **Image quality differs in kind.** Ours has the most out-of-focus fields (11%), Radboud the most pen ink (6% of fields), Karolinska the sharpest images and two scanners at two pixel sizes; SICAPv2 is distributed without a stated pixel size, so its scale is an estimate (1.75x ours, from nucleus size).

Numbers: <dataset>_report.md, comparison_table.tex; figures fig_stain.pdf, fig_label_geometry.pdf, fig_<dataset>_samples.pdf, fig_domain_distance.pdf.
