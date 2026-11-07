"""t̅t + J/ψ -> μμ, the search for associated production (CMS Run 3, 2022-2026): the analysis told in one animation.
Manim Community 0.19 (Text / MarkupText only, no LaTeX). The numbers come from the deck's JSON files next to this directory
(splot.json, mc_compare.json, mc_template.json; TTJ_FILES overrides the directory).
Beats: title; a l + jets t̅t event in the eta-phi plane; three J/ψ and the veto of the top-decay jets; the decay length
l_xy and the fit; the twist the t̅t simulation showed (short b decays in untagged b jets) and its closure; data against
the simulation.
render (the anim environment of pixi.toml; from this directory):
    pixi run -e anim manim render -r 1920,1080 --fps 30 --format mp4 -o ttjpsi_story ttjpsi_story.py TtJpsiStory
    (preview: pixi run -e anim manim render -ql ttjpsi_story.py TtJpsiStory)"""
import json
import os

import numpy as np
from manim import (DOWN, LEFT, ORIGIN, RIGHT, UP, Arrow, Axes, Circle, Create, DashedLine, DashedVMobject, Dot,
                   FadeIn, FadeOut, GrowArrow, GrowFromCenter, Indicate, Line, MarkupText, Rectangle, Scene,
                   SurroundingRectangle, Text, Transform, ValueTracker, VGroup, Write, always_redraw, config, rate_functions)

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = os.environ.get("TTJ_FILES", os.path.dirname(HERE))


def J(name):
    return json.load(open(os.path.join(FILES, name)))


SPL, MCC, MTF = J("splot.json"), J("mc_compare.json"), J("mc_template.json")
NP, NPE, SIG = SPL["fits"]["notb"]["N"][0], SPL["fits"]["notb"]["N_err"][0], SPL["fits"]["notb"]["signif_prompt"]
NNP = SPL["fits"]["notb"]["N"][1]
CS, CT = MCC["closure"]["signal"], MCC["closure"]["ttj"]
MC_P = MCC["rates"]["signal_all"]["mc_prompt"][0]
TTJ_C = MTF["fits"]["ttj_ljets_nbcon"]

BG, INK, INK2, GREY = "#F7F6F2", "#14213D", "#3A4556", "#9AA3AD"
B, P, BLUE, TOT, EXTRA, PINK = "#eb6834", "#1baf7a", "#2a78d6", "#4a3aa7", "#eda100", "#e87ba4"
FONT = "DejaVu Sans"
config.background_color = BG
TT = "t̅t"


def T(s, size=30, color=INK, bold=False):
    return Text(s, font=FONT, font_size=size, color=color, weight="BOLD" if bold else "NORMAL")


def M(s, size=30, color=INK):
    return MarkupText(s, font=FONT, font_size=size, color=color)


class TtJpsiStory(Scene):
    def construct(self):
        self.title_card()
        self.event()
        self.lifetime()
        self.simulation()
        self.closing()

    # ------------------------------------------------------------------ 1. title
    def title_card(self):
        t1 = T(f"{TT} + J/ψ → μμ", 64, INK, bold=True)
        t2 = T("the J/ψ that are not from the top decays", 40, INK2)
        t3 = T("CMS Run 3 · 2022–2026 · 310 fb⁻¹ · √s = 13.6 TeV", 26, GREY)
        g = VGroup(t1, t2, t3).arrange(DOWN, buff=0.45)
        self.play(FadeIn(t1, shift=0.3 * UP), run_time=1.2)
        self.play(FadeIn(t2), run_time=1.0)
        self.play(FadeIn(t3), run_time=0.8)
        self.wait(2.2)
        self.play(FadeOut(g), run_time=0.8)

    # ------------------------------------------------------------------ 2-3. the event and the veto
    def event(self):
        head = T(f"a {TT} event, ℓ + jets", 34, INK, bold=True).to_edge(UP, buff=0.35)
        # eta-phi plane, one scale for both axes so that Delta R = 0.4 is a circle
        k = 0.95
        ax = Axes(x_range=[-2.5, 2.5, 1], y_range=[-np.pi, np.pi, np.pi / 2], x_length=5 * k, y_length=2 * np.pi * k,
                  axis_config=dict(color=INK2, stroke_width=2, include_tip=False, include_ticks=False))
        frame = Rectangle(width=5 * k, height=2 * np.pi * k, color=INK2, stroke_width=2)
        plane = VGroup(frame).move_to(3.6 * LEFT + 0.35 * DOWN)
        ax.move_to(plane.get_center())
        xl = T("η", 30, INK2).next_to(frame, DOWN, buff=0.15)
        yl = T("φ", 30, INK2).next_to(frame, LEFT, buff=0.15)
        self.play(FadeIn(head), Create(frame), FadeIn(xl), FadeIn(yl), run_time=1.2)

        def jet(eta, phi, col, lab):
            c = Circle(radius=0.4 * k, color=col, stroke_width=4).move_to(ax.c2p(eta, phi))
            c.set_fill(col, opacity=0.18)
            rng = np.random.default_rng(int(abs(eta * 100 + phi * 37)))
            parts = VGroup(*[Dot(ax.c2p(eta + r * np.cos(a), phi + r * np.sin(a)), radius=0.025, color=col)
                             for r, a in zip(0.3 * rng.random(9), 2 * np.pi * rng.random(9))])
            lb = T(lab, 26, col, bold=True).next_to(c, UP, buff=0.05)
            return VGroup(c, parts, lb)

        # the decay chain, written out on the right while its pieces appear in the plane
        x0 = 1.0
        chain = [
            (M(f"t → W<sup>+</sup> b", 32), "b"),
            (M("W<sup>+</sup> → ℓ<sup>+</sup> ν", 32), "lep"),
            (M(f"t̅ → W<sup>−</sup> b̅", 32), "bbar"),
            (M("W<sup>−</sup> → q q̅′", 32), "w"),
        ]
        COLX = -0.4  # left edge of the text column right of the plane
        rows = VGroup(*[c for c, _ in chain]).arrange(DOWN, aligned_edge=LEFT, buff=0.42)
        rows.move_to(np.array([0, 1.45, 0])).align_to(np.array([COLX, 0, 0]), LEFT)
        jb = jet(-0.9, 1.6, B, "b")
        jbb = jet(1.1, -1.2, B, "b̅")
        jq1 = jet(0.4, 1.9, BLUE, "q")
        jq2 = jet(1.6, 0.6, BLUE, "q̅′")
        lep = VGroup(Dot(ax.c2p(-1.6, -0.4), radius=0.09, color=TOT), T("ℓ", 28, TOT, bold=True).next_to(ax.c2p(-1.6, -0.4), RIGHT, buff=0.1))
        met = VGroup(DashedLine(ax.c2p(-2.5, -2.3), ax.c2p(2.5, -2.3), color=GREY, stroke_width=3, dash_length=0.12),
                     T("missing momentum (its φ)", 20, GREY).next_to(ax.c2p(-2.45, -2.3), UP, buff=0.05, aligned_edge=LEFT))
        pieces = {"b": [jb], "lep": [lep, met], "bbar": [jbb], "w": [jq1, jq2]}
        for row, key in chain:
            self.play(Write(row), run_time=0.9)
            self.play(*[FadeIn(p, scale=0.6) for p in pieces[key]], run_time=0.9)
            self.wait(0.5)
        self.wait(1.0)

        # three J/psi -> mu mu
        def jpsi(eta, phi, col):
            d = Dot(ax.c2p(eta, phi), radius=0.08, color=col)
            m1 = Line(ax.c2p(eta, phi), ax.c2p(eta + 0.22, phi + 0.16), color=col, stroke_width=3)
            m2 = Line(ax.c2p(eta, phi), ax.c2p(eta + 0.2, phi - 0.17), color=col, stroke_width=3)
            return VGroup(d, m1, m2)

        j_b, j_w, j_s = jpsi(-0.95, 1.5, INK), jpsi(1.55, 0.7, INK), jpsi(-0.6, -1.6, INK)
        sub = T("three J/ψ → μμ", 32, INK, bold=True).move_to(np.array([0, -0.9, 0])).align_to(np.array([COLX, 0, 0]), LEFT)
        self.play(FadeIn(sub), *[GrowFromCenter(j) for j in (j_b, j_w, j_s)], run_time=1.2)
        self.wait(1.2)
        l1 = M("in a b jet: <span foreground='#eb6834'>b-hadron decay</span>", 26, INK2)
        l2 = M("in a W jet: <span foreground='#2a78d6'>W decay</span>", 26, INK2)
        l3 = M("outside every jet", 26, INK2)
        legs = VGroup(l1, l2, l3).arrange(DOWN, aligned_edge=LEFT, buff=0.25).next_to(sub, DOWN, buff=0.35).align_to(sub, LEFT)
        for j, l in ((j_b, l1), (j_w, l2), (j_s, l3)):
            self.play(Indicate(j, color=P, scale_factor=1.8), FadeIn(l), run_time=0.9)
        self.wait(1.5)

        # the veto: every jet of the two top decays
        veto = VGroup(T("veto: ΔR < 0.4 from every jet", 28, INK, bold=True), T("of the two top decays (dashed)", 28, INK, bold=True)).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        veto.move_to(np.array([0, 1.9, 0])).align_to(np.array([COLX, 0, 0]), LEFT)
        cones = VGroup(*[DashedVMobject(Circle(radius=0.4 * k, color=INK, stroke_width=3).move_to(j[0].get_center()), num_dashes=18)
                         for j in (jb, jbb, jq1, jq2)])
        self.play(FadeOut(rows), FadeIn(veto), *[Create(c) for c in cones], VGroup(sub, legs).animate.shift(1.25 * UP), run_time=1.4)
        self.wait(1.0)
        cross = lambda m: VGroup(Line(m.get_corner(UP + LEFT), m.get_corner(DOWN + RIGHT), color=GREY, stroke_width=4),  # noqa: E731
                                 Line(m.get_corner(DOWN + LEFT), m.get_corner(UP + RIGHT), color=GREY, stroke_width=4))
        x_b, x_w = cross(SurroundingRectangle(j_b, buff=0.06)), cross(SurroundingRectangle(j_w, buff=0.06))
        v1 = T("vetoed: top decay products", 28, GREY, bold=True).next_to(legs, DOWN, buff=0.4).align_to(legs, LEFT)
        self.play(j_b.animate.set_color(GREY), j_w.animate.set_color(GREY), Create(x_b), Create(x_w), l1.animate.set_opacity(0.45),
                  l2.animate.set_opacity(0.45), FadeIn(v1), run_time=1.4)
        self.wait(1.2)
        halo = Circle(radius=0.32, color=P, stroke_width=5).move_to(j_s[0].get_center())
        v2 = T("signal: associated production", 28, P, bold=True).next_to(v1, DOWN, buff=0.25).align_to(v1, LEFT)
        self.play(j_s.animate.set_color(P), Create(halo), l3.animate.set_color(P), FadeIn(v2), run_time=1.4)
        self.wait(3.0)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.0)

    # ------------------------------------------------------------------ 4. decay length and the fit
    def lifetime(self):
        head = M("prompt or b decay? the decay length ℓ<sub>xy</sub>", 34, INK).to_edge(UP, buff=0.35)
        self.play(FadeIn(head), run_time=0.8)
        # transverse plane, not to scale
        pv = Dot(np.array([-5.4, 0.4, 0]), radius=0.1, color=INK)
        pvl = T("primary vertex", 22, INK2).next_to(pv, DOWN, buff=0.15)
        self.play(FadeIn(pv), FadeIn(pvl), run_time=0.6)
        # prompt: the muons start at the vertex
        mu1 = Arrow(pv.get_center(), pv.get_center() + np.array([2.6, 1.7, 0]), color=P, buff=0, stroke_width=5, max_tip_length_to_length_ratio=0.08)
        mu2 = Arrow(pv.get_center(), pv.get_center() + np.array([2.8, 0.9, 0]), color=P, buff=0, stroke_width=5, max_tip_length_to_length_ratio=0.08)
        pl = M("prompt J/ψ: ℓ<sub>xy</sub> ≈ 0", 26, P).next_to(mu1.get_end(), RIGHT, buff=0.15)
        self.play(GrowArrow(mu1), GrowArrow(mu2), run_time=0.9)
        self.play(FadeIn(pl), run_time=0.6)
        self.wait(0.8)
        # b decay: the b hadron flies, then decays
        sv = Dot(pv.get_center() + np.array([1.6, -1.5, 0]), radius=0.08, color=B)
        fl = DashedLine(pv.get_center(), sv.get_center(), color=B, stroke_width=4)
        bl = T("b hadron", 22, B).next_to(fl.get_center(), LEFT, buff=0.15)
        mu3 = Arrow(sv.get_center(), sv.get_center() + np.array([2.4, 0.2, 0]), color=B, buff=0, stroke_width=5, max_tip_length_to_length_ratio=0.08)
        mu4 = Arrow(sv.get_center(), sv.get_center() + np.array([2.3, -0.8, 0]), color=B, buff=0, stroke_width=5, max_tip_length_to_length_ratio=0.08)
        self.play(Create(fl), FadeIn(bl), run_time=1.0)
        self.play(FadeIn(sv), GrowArrow(mu3), GrowArrow(mu4), run_time=0.9)
        bdl = M("b decay: ℓ<sub>xy</sub> = L · p̂<sub>T</sub>(J/ψ) ≈ 0.4 mm", 26, B).next_to(mu4.get_end(), DOWN, buff=0.15).shift(1.2 * LEFT)
        self.play(FadeIn(bdl), run_time=0.7)
        self.wait(2.0)
        sketch = VGroup(pv, pvl, mu1, mu2, pl, sv, fl, bl, mu3, mu4, bdl)
        self.play(sketch.animate.scale(0.62).to_edge(LEFT, buff=0.3).shift(0.4 * UP), run_time=1.0)

        # the l_xy distribution of the J/psi outside the ttbar jets: prompt peak + b-decay tail (shapes of the fit)
        ax = Axes(x_range=[-0.3, 1.2, 0.3], y_range=[0, 1.15, 0.5], x_length=7.4, y_length=4.0,
                  axis_config=dict(color=INK2, stroke_width=2, include_tip=False, include_ticks=True, tick_size=0.06))
        ax.move_to(np.array([2.6, -0.05, 0]))
        ticks = VGroup(*[T(f"{v:g}", 20, INK2).next_to(ax.c2p(v, 0), DOWN, buff=0.12) for v in (0, 0.3, 0.6, 0.9)])
        xl = M("ℓ<sub>xy</sub> [mm]", 24, INK2).next_to(ax.x_axis, DOWN, buff=0.45).align_to(ax.x_axis, RIGHT)
        yl = T("J/ψ", 24, INK2).next_to(ax.y_axis, UP, buff=0.1)
        edges = np.linspace(-0.3, 1.2, 31)
        c = 0.5 * (edges[1:] + edges[:-1])
        w = edges[1] - edges[0]
        s_res, lam = 0.04, 0.38
        gp = np.exp(-0.5 * (c / s_res) ** 2)
        gp = gp / gp.sum() * NP
        dec = np.where(c > 0, np.exp(-np.maximum(c, 0) / lam), 0.0)
        dec = np.convolve(dec, np.exp(-0.5 * (np.arange(-3, 4) * w / s_res) ** 2), mode="same")
        dec = dec / dec.sum() * NNP
        top = (gp + dec).max()
        hp, hd = gp / top, dec / top
        grow = ValueTracker(0.0)

        def bars():
            f = grow.get_value()
            g = VGroup()
            for x, a, b_ in zip(c, hp, hd):
                y0 = ax.c2p(x, 0)[1]
                yb = ax.c2p(x, f * b_)[1]
                ya = ax.c2p(x, f * (a + b_))[1]
                xl_, xr_ = ax.c2p(x - 0.45 * w, 0)[0], ax.c2p(x + 0.45 * w, 0)[0]
                if yb - y0 > 1e-3:
                    g.add(Rectangle(width=xr_ - xl_, height=yb - y0, stroke_width=0, fill_color=B, fill_opacity=0.75).move_to([(xl_ + xr_) / 2, (y0 + yb) / 2, 0]))
                if ya - yb > 1e-3:
                    g.add(Rectangle(width=xr_ - xl_, height=ya - yb, stroke_width=0, fill_color=P, fill_opacity=0.85).move_to([(xl_ + xr_) / 2, (yb + ya) / 2, 0]))
            return g

        hist = always_redraw(bars)
        self.play(Create(ax), FadeIn(ticks), FadeIn(xl), FadeIn(yl), run_time=1.0)
        self.add(hist)
        self.play(grow.animate.set_value(1.0), run_time=3.5, rate_func=rate_functions.ease_out_cubic)
        self.wait(0.5)
        k1 = VGroup(Rectangle(width=0.35, height=0.22, fill_color=P, fill_opacity=0.85, stroke_width=0), T("prompt J/ψ", 22, P)).arrange(RIGHT, buff=0.15)
        k2 = VGroup(Rectangle(width=0.35, height=0.22, fill_color=B, fill_opacity=0.75, stroke_width=0), T("b-hadron decays", 22, B)).arrange(RIGHT, buff=0.15)
        key = VGroup(k1, k2).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(ax.c2p(0.42, 1.05), RIGHT, buff=0)
        fitl = M("unbinned fit of m(μμ) × ℓ<sub>xy</sub>", 22, INK2).next_to(key, DOWN, buff=0.2).align_to(key, LEFT)
        self.play(FadeIn(key), FadeIn(fitl), run_time=0.8)
        self.wait(1.5)
        res = T(f"{NP:.0f} ± {NPE:.0f} prompt J/ψ outside the {TT} jets, {SIG:.1f}σ", 34, P, bold=True).to_edge(DOWN, buff=0.3)
        box = SurroundingRectangle(res, color=P, buff=0.15, corner_radius=0.08)
        self.play(Write(res), Create(box), run_time=1.6)
        self.wait(3.0)
        self.remove(hist)
        self.add(bars())  # the final histogram, static
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.0)

    # ------------------------------------------------------------------ 5. the simulation
    def simulation(self):
        head = T(f"the CMS {TT} simulation, through the same fit", 34, INK, bold=True).to_edge(UP, buff=0.35)
        self.play(FadeIn(head), run_time=0.8)
        # left: an untagged b jet, a short flight
        pv = Dot(np.array([-5.6, 1.2, 0]), radius=0.09, color=INK)
        sv = Dot(pv.get_center() + np.array([0.45, -0.25, 0]), radius=0.07, color=B)
        fl = DashedLine(pv.get_center(), sv.get_center(), color=B, stroke_width=4)
        m1 = Arrow(sv.get_center(), sv.get_center() + np.array([2.3, 0.6, 0]), color=B, buff=0, stroke_width=5, max_tip_length_to_length_ratio=0.08)
        m2 = Arrow(sv.get_center(), sv.get_center() + np.array([2.3, -0.3, 0]), color=B, buff=0, stroke_width=5, max_tip_length_to_length_ratio=0.08)
        t1 = T("a b jet that failed the b tag:", 26, INK).move_to(np.array([-3.6, 2.35, 0]))
        t2 = T("the b hadron flew little", 26, B).next_to(t1, DOWN, buff=0.15).align_to(t1, LEFT)
        self.play(FadeIn(t1), FadeIn(pv), run_time=0.8)
        self.play(Create(fl), FadeIn(sv), GrowArrow(m1), GrowArrow(m2), FadeIn(t2), run_time=1.2)
        t3 = M("short ℓ<sub>xy</sub>: looks prompt", 26, P).next_to(m2.get_end(), DOWN, buff=0.3).shift(1.4 * LEFT)
        self.play(FadeIn(t3), run_time=0.7)
        self.wait(1.5)

        # right: the b-decay l_xy shapes, tagged b jets against the vetoed (untagged) jets
        ax = Axes(x_range=[-0.2, 1.0, 0.2], y_range=[0, 1.1, 0.5], x_length=6.0, y_length=3.2,
                  axis_config=dict(color=INK2, stroke_width=2, include_tip=False, include_ticks=True, tick_size=0.05))
        ax.move_to(np.array([3.4, 1.15, 0]))
        ticks = VGroup(*[T(f"{v:g}", 18, INK2).next_to(ax.c2p(v, 0), DOWN, buff=0.1) for v in (0, 0.4, 0.8)])
        xl = M("ℓ<sub>xy</sub> [mm]", 22, INK2).next_to(ax.x_axis, DOWN, buff=0.4).align_to(ax.x_axis, RIGHT)
        long_ = ax.plot(lambda x: (np.exp(-max(x, 0) / 0.38) * (1 - np.exp(-max(x, 0) / 0.03))) / 0.9, x_range=[-0.2, 1.0, 0.005], color=TOT, stroke_width=4)
        short = ax.plot(lambda x: np.exp(-0.5 * ((x - 0.03) / 0.045) ** 2) * (1.0 if x > -0.1 else 0.0), x_range=[-0.2, 1.0, 0.005], color=EXTRA, stroke_width=4)
        c1 = T("b jets", 22, TOT).next_to(ax.c2p(0.45, 0.55), RIGHT, buff=0.05)
        c2 = T("vetoed jets (b tag failed)", 22, EXTRA).next_to(ax.c2p(0.12, 0.95), RIGHT, buff=0.05)
        self.play(Create(ax), FadeIn(ticks), FadeIn(xl), run_time=0.8)
        self.play(Create(long_), FadeIn(c1), run_time=1.0)
        self.play(Create(short), FadeIn(c2), run_time=1.0)
        self.wait(1.5)
        r1 = T(f"vetoed jets: the fit finds {CT['fit_prompt']:.0f} prompt J/ψ for {CT['truth_prompt']:.0f} true", 28, EXTRA, bold=True)
        r2 = T(f"refitted with their own shape: {TTJ_C['N_p']:.0f} ± {TTJ_C['N_p_err']:.0f} of the 76 in data", 26, INK2)
        r3 = T(f"outside the {TT} jets it closes: {CS['fit_prompt']:.0f} ± {CS['fit_prompt_err']:.0f} found for {CS['truth_prompt']:.0f} true", 28, P, bold=True)
        rs = VGroup(r1, r2, r3).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(np.array([0, -2.55, 0]))
        self.play(FadeIn(r1), run_time=0.8)
        self.wait(1.2)
        self.play(FadeIn(r2), run_time=0.8)
        self.wait(1.2)
        self.play(FadeIn(r3), run_time=0.8)
        self.wait(3.0)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.0)

    # ------------------------------------------------------------------ 6. data against the simulation
    def closing(self):
        head = T(f"prompt J/ψ outside the {TT} jets", 36, INK, bold=True).to_edge(UP, buff=0.5)
        self.play(FadeIn(head), run_time=0.8)
        base = -1.0
        scale = 3.3 / 115.0
        bars = VGroup()
        labs = VGroup()
        for i, (val, err, col, nm) in enumerate(((NP, NPE, P, "data"), (MC_P, 0.0, GREY, f"{TT} simulation\n(hadronisation only)"))):
            x = -1.6 + 3.2 * i
            r = Rectangle(width=1.4, height=max(val * scale, 0.02), fill_color=col, fill_opacity=0.85, stroke_width=0)
            r.move_to(np.array([x, base + val * scale / 2, 0]))
            bars.add(r)
            num = T(f"{val:.0f} ± {err:.0f}" if err else f"{val:.0f}", 32, col, bold=True).next_to(r, UP, buff=0.15)
            lab = T(nm, 22, INK2).next_to(np.array([x, base, 0]), DOWN, buff=0.2)
            labs.add(VGroup(num, lab))
        floor = Line(np.array([-3.0, base, 0]), np.array([3.0, base, 0]), color=INK2, stroke_width=2)
        self.play(Create(floor), run_time=0.5)
        for r, l in zip(bars, labs):
            self.play(FadeIn(r, shift=0.3 * UP), FadeIn(l), run_time=1.0)
        self.wait(1.5)
        q = T("a second parton scattering, or J/ψ from the t̅t shower?", 30, INK, bold=True).to_edge(DOWN, buff=0.95)
        n = T("next: generators and tunes (CR1, CR2, ERD, Sherpa, MadGraph)", 24, INK2).next_to(q, DOWN, buff=0.25)
        self.play(FadeIn(q), run_time=0.9)
        self.play(FadeIn(n), run_time=0.8)
        self.wait(4.0)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.2)
        self.wait(0.3)
