# Project rules

- Decks are HTML with live plotlyhep figures, no PNG on the slides, no claude.ai copies -- the user's deck format.
- Report plots by their deck link, not by describing them ("you can just tell me the link").
- Selection, normalisation and category go in the legend title, never on the canvas; keep the title to two
  short lines (`<br>`): a long title squeezes the plot area in plotly.
- Window - sidebands histograms: zero line and a lower y limit below the lowest point.
- Log-x axes: explicit tick values (plotly otherwise labels 20, 30 as "2, 3").
