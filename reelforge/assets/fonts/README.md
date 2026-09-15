# Fonts

Arabic captions need a font with Arabic coverage — most default sans fonts have none, and
libass will silently draw empty boxes instead of letters.

Fetch them with:

    reelforge setup

That downloads Cairo, Tajawal Bold and Almarai ExtraBold (all SIL Open Font License) into
this folder. They are not committed to the repo, only downloaded on demand.

Any `.ttf`/`.otf` you drop in here is picked up too — set `captions.font` to its family name.
