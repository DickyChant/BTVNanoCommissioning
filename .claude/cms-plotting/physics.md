# Physics

## Process

- ttbar + J/psi -> mu mu, CMS Run 3 (2022-2026, 13.6 TeV, about 310 fb^-1); channels mu + jets, e + jets, e mu.
- Signal: prompt J/psi produced with the ttbar (associated production), required Delta R > 0.4 from every jet of
  the two top decays. Categories: b jet (cat 0), other ttbar jet (jcat 1, vetoed), outside the ttbar jets (jcat >= 2, signal).
- Simulation: Summer24 TTtoLNu2Q + TTto2L2Nu (POWHEG + Pythia 8 CP5, no onia in the shower: its J/psi are b decays,
  plus a few from low-mass c c-bar strings); MC truth 0 prompt, 1 b-hadron ancestor, 2 no common J/psi parent.
  Private gen-level Pythia 8 + LETO (onia in the shower, LDMEs x 20) for the prompt rate.

## Observables

| observable | x label | range, bins | what is in it |
|---|---|---|---|
| m(mu mu) | m(μ⁺μ⁻) [GeV] | 2.6-3.6 GeV | J/psi peak, sigma ~ 30 MeV; window 2.95-3.25, sidebands 2.60-2.90 and 3.30-3.60 |
| l_xy | ℓ_xy [mm] | -0.5-1.5 mm, variable | prompt core (resolution ~ 20 um x S), b-decay tail (lambda ~ 0.38 mm); window - sidebands goes negative: zero line |
| pT(J/psi) | p_T(J/ψ) [GeV] | 3-100 GeV, variable | signal soft (~ 10 GeV), inclusive-like |

The J/psi in an untagged b jet (the vetoed category) have short decay lengths (the tag failed because the
b hadron flew little): the b-jet l_xy template does not describe them.

## Fits

- Unbinned m x l_xy (sPlot): Gaussian-like J/psi mass x {prompt G(0, S err), b decay exp(lambda) x G, continuum};
  b-decay template of the non-b categories from the b-jet fit (lambda, prompt-like share FPI).
- Weights: prompt / non-prompt sWeights; errors sqrt(sum w^2).

## Blind or sensitive

Nothing blind.
