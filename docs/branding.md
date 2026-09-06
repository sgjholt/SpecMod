# SpecMod Branding Manual

Defines the visual identity, typography standards, and documentation theme for SpecMod. Target aesthetic: crisp and academic, resembling a published journal layout.

---

## 1. Visual Identity

### 1.1 Colour Palette

**Light mode**

| Role | Hex | Name |
| --- | --- | --- |
| Page background | `#FDFBF7` | Warm Manuscript Off-White |
| Page background (alt) | `#FFFFFF` | Pure Paper White |
| Body text | `#1E293B` | Ink Slate |
| Primary accent | `#1E40AF` | Journal Navy |
| Secondary accent / plot line | `#C2410C` | Terracotta Red |
| Code block background | `#F8FAFC` | — |
| Code block border | `#E2E8F0` | — |

**Dark mode**

| Role | Hex | Name |
| --- | --- | --- |
| Page background | `#0F172A` | Deep Slate |
| Body text | `#F8FAFC` | Paper White |
| Primary accent | `#60A5FA` | Soft Navy |
| Secondary accent / plot line | `#FB923C` | Soft Terracotta |
| Code block background | `#1E293B` | — |
| Code block border | `#334155` | — |

**Usage rules**

- Accent colours are reserved for data highlights and interactive elements (links, active states, buttons, spectral model curves).
- Backgrounds remain neutral.
- Primary accent marks navigation and interaction; secondary accent marks fitted model curves against observed data.

### 1.2 Logo System

**Concept:** minimalist line-art spectral window enclosing a seismic waveform that flattens into a model curve.

**Colour:** monochromatic `#1E293B` with a single accent — either Terracotta Red or Journal Navy.

**Variants**

1. **Horizontal logo** — logo mark plus "SpecMod" set in the heading serif. Use in page headers.
2. **Square icon / favicon** — waveform mark only. Use for browser tabs and avatars.

**Usage rules**

- One accent colour per logo instance.
- Maintain clear space on all sides equal to the height of the logo mark.

```
[Horizontal Logo Here]       [Square Icon Here]
```

---

## 2. Typography

### 2.1 Fonts

| Role | Primary | Fallback |
| --- | --- | --- |
| Headings (H1–H6) | Merriweather (serif) | Georgia, serif |
| Body and navigation | Open Sans (sans-serif) | system sans |
| Code and inline syntax | Fira Code | Computer Modern Typewriter, monospace |

### 2.2 Usage Rules

- Headings always use the serif stack.
- Body text uses the sans-serif stack for on-screen legibility.
- Inline code and maths are styled distinctly to emulate technical papers.

```
H1:   Spectral Modeling Overview      Merriweather, 32px
Body: SpecMod processes seismic...    Open Sans, 16px
Code: model.fit_spectrum(data)        Fira Code, 14px
```

---

## 3. Documentation Theme

### 3.1 Sphinx Configuration — `conf.py`

```python
html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["academic.css"]

html_theme_options = {
    "light_css_variables": {
        "color-background-primary": "#FDFBF7",
        "color-background-secondary": "#F8FAFC",
        "color-background-border": "#E2E8F0",
        "color-foreground-primary": "#1E293B",
        "color-brand-primary": "#1E40AF",
        "color-brand-content": "#C2410C",
        "font-stack": "'Open Sans', sans-serif",
        "font-stack--monospace": "'Fira Code', 'Computer Modern Typewriter', monospace",
    },
    "dark_css_variables": {
        "color-background-primary": "#0F172A",
        "color-background-secondary": "#1E293B",
        "color-background-border": "#334155",
        "color-foreground-primary": "#F8FAFC",
        "color-brand-primary": "#60A5FA",
        "color-brand-content": "#FB923C",
    },
    "light_logo": "specmod-logo-academic.png",
    "dark_logo": "specmod-logo-academic-dark.png",
    "sidebar_hide_name": True,
}
```

Font stacks are declared once under `light_css_variables`; furo emits those on `body` as the base declaration, so they carry into dark mode without duplication.

### 3.2 Custom Stylesheet — `_static/academic.css`

```css
@import url('https://fonts.googleapis.com/css2?family=Fira+Code&family=Merriweather:wght@400;700&family=Open+Sans:wght@400;600&display=swap');

h1, h2, h3, h4, h5, h6 {
    font-family: 'Merriweather', Georgia, serif !important;
    color: var(--color-foreground-primary);
}

div.highlight {
    background-color: var(--color-background-secondary) !important;
    border: 1px solid var(--color-background-border);
    border-radius: 4px;
    padding: 0.75em;
    overflow-x: auto;
}

code, pre {
    font-family: 'Fira Code', 'Computer Modern Typewriter', monospace;
    font-size: 0.875rem;
    color: var(--color-foreground-primary);
}

a {
    color: var(--color-brand-primary);
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}
```

---

## 4. Implementation Notes

- Keep design changes consistent with the academic aesthetic: neutral backgrounds, limited accents.
- Test Sphinx builds in both light and dark modes before release.
- Version `_static` assets for cache control.
- All colour values are referenced through CSS custom properties, never hard-coded in the stylesheet, so a palette change is a single edit in `conf.py`.
