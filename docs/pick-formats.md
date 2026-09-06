# Reading picks, and adding a format

`specmod.preprocess.with_picks` returns a copy of a stream with P and S
arrivals attached:

```python
import specmod.preprocess as pre

stream = pre.with_picks(stream, "event.xml")
```

The format is detected from the file, not from its name. This page covers what
is read out of the box, what to do when your format is not, and how the
matching between a pick and a trace works.

## What is read out of the box

Two readers ship:

| Reader | Formats |
|---|---|
| `obspy_events` | everything `obspy.read_events` parses — QuakeML, SEISAN Nordic, HypoDD `.pha`, NonLinLoc `.hyp`, IMS/GSE bulletins, and more |
| `snuffler` | Snuffler / Pyrocko marker files |

Measured, by writing each format and reading it back: **QuakeML, Nordic and
HypoDD** carry their picks through intact. **SC3ML/SCML does not** — ObsPy
1.5.0 writes the `<pick>` elements and drops them on read, so SeisComP output
does not work today. NonLinLoc and the bulletins are expected to work but are
not yet confirmed against a real file.

Detection requires exactly one reader to claim a file. If none does, the error
lists what was tried; if several do, that is a defect in their sniffing and the
error says so. Either way you can force one:

```python
stream = pre.with_picks(stream, "picks.txt", format="obspy_events")
```

## If your format is an event file, register it with ObsPy

**This is the route to prefer.** ObsPy's own plugin system takes a format
through entry points, and a format registered there is readable by
`obspy.read_events` — which means SpecMod reads it through the `obspy_events`
delegate **with no SpecMod-side registration at all**, and the same parser
serves every other ObsPy-based tool you run.

```toml
[project.entry-points."obspy.plugin.event"]
MYFORMAT = "my_package.io"

[project.entry-points."obspy.plugin.event.MYFORMAT"]
isFormat = "my_package.io:_is_myformat"
readFormat = "my_package.io:_read_myformat"
```

Nothing else is needed. Detection here asks ObsPy's own `isFormat` functions,
so your format joins the roster the moment it is installed.

## If it is not an event file

Picker output, associator output and in-house arrival tables are not
catalogues and do not belong in ObsPy's roster. Register those with SpecMod.

### A delimited table

For one-row-per-arrival tables. You supply the column mapping, because there is
no standard to assume:

```python
import specmod.picks as pk

reader = pk.CSVPickReader(
    columns={
        "station": "station",       # required
        "phase": "phase_type",      # required
        "time": "arrival_time",     # required
        "network": "net",
        "location": "loc",
        "channel": "chan",
        "weight": "probability",
    },
    reader_name="phasenet",
)
pk.register_reader(reader)

stream = pre.with_picks(stream, "picks.csv")   # detected by its column headings
```

Four classes, differing only in how a line is split:

| Class | Separator | Suffixes |
|---|---|---|
| `CSVPickReader` | `,`, with `csv` quoting | `.csv` |
| `TSVPickReader` | a tab | `.tsv`, `.tab` |
| `WhitespacePickReader` | runs of spaces or tabs | `.txt`, `.dat`, `.lst` |
| `DelimitedPickReader` | whatever you pass as `delimiter` | whatever you pass as `file_suffixes` |

`DelimitedPickReader` is the general case — the other three are it with the
delimiter and the plausible suffixes already set, so `pk.TSVPickReader(columns=…)`
is all a tab-separated table needs. Use the base directly for anything else:

```python
pk.DelimitedPickReader(
    columns={...}, reader_name="piped", delimiter="|", file_suffixes=(".dat",)
)
```

`delimiter` is one character, or `None` for whitespace. The whitespace path is
a separate parse — quoting has no meaning in a space-aligned table, and cells
cannot contain spaces.

One overlap to know about: **whitespace-splitting subsumes tab-splitting**, so
`WhitespacePickReader` claims a `.tsv` too. Registering both against the same
column names makes every tab-separated file ambiguous. Register one, or pass
`format=`.

Mappable fields are `station`, `phase`, `time`, `network`, `location`,
`channel`, `weight`, `uncertainty`, `polarity` and `author`. Phase values fold
to `P` or `S` on the first letter, with the original kept as `raw_phase`; rows
that are neither are skipped.

**No presets ship for PhaseNet, EQTransformer or SeisBench.** Each writes
several column layouts across versions, and a preset written from
documentation rather than from a real output file is a guess with a name on it.
Write the four lines above against the file you actually have.

Mapping the `location` column matters more than it looks. A column that is
present but blank states an *empty* location code; a column you do not map at
all states *nothing*, and the pick will then match any location code at that
station — see matching, below.

### Your own reader

Anything satisfying the `PickReader` protocol can be registered:

```python
class MyReader:
    name = "my_format"
    suffixes = (".arr",)

    def can_read(self, source) -> bool: ...
    def read(self, source) -> list[PickSet]: ...
```

`can_read` must be cheap — read a header, not the file — and **must not raise**
on a missing, empty, truncated or binary file, or on any other format's files.
`read` returns one `PickSet` per event in the file.

For a reader in an installed package, advertise it instead of calling
`register_reader`:

```toml
[project.entry-points."specmod.pick_readers"]
my_format = "my_package.picks:MyReader"
```

The entry point names a zero-argument callable returning the reader. Discovery
happens on first use of the registry, so an installed plugin costs nothing to a
caller who never reads a pick. A plugin that fails to import warns, naming its
distribution, and is skipped; it cannot make the built-in formats unreadable,
and it cannot take a built-in name.

`tests/test_pick_readers.py::TestReaderContract` is parameterised over every
registered reader — run it against yours.

## How a pick reaches a trace

Picks are matched **per sensor** — `NET.STA.LOC` — not per channel. An arrival
is one sensor's observation, and the component it happened to be picked on is
incidental: on a typical file every station has P on `HHZ` and S on `HHN`, so
matching on the full SEED id would leave the horizontals with no S at all.

Most formats supply less than a full identity — a bare station code is
common. A pick matches a trace when every field it **states** agrees, so a
station-only pick still reaches its sensor. Then:

- **one match** — attached;
- **no match** — the pick is unused, and counted in the report;
- **several matches** — an error naming the candidates.

The last case is a station with two instruments, typically a surface and a
borehole one, which differ only by location code and do not see the same
arrival. Broadcasting a station-only pick to both is never silently correct, so
you have to say what you want:

```python
stream = pre.with_picks(stream, "picks.csv", on_ambiguous="broadcast")  # co-located
stream = pre.with_picks(stream, "picks.csv", on_ambiguous="skip")       # drop them
```

### Several picks for one arrival

Reduced by a policy, rather than by whichever came last in the file:
`prefer_reviewed` (the default — an analyst's pick over an automatic one, then
earliest), `earliest`, `highest_weight`, or `error` to refuse to choose.

### Several events in one file

Bulletins hold many events, and merging their arrivals describes none of them.
A multi-event source raises unless you select:

```python
stream = pre.with_picks(stream, "bulletin.ims", event_id="smi:local/12345")
```

### What was actually attached

```python
report = []
stream = pre.with_picks(stream, "picks.csv", report=report)
print(report[0].summary())
# 28 attached to 14 sensors, 2 unused, 0 ambiguous, 0 resolved by policy
```

`unused` is the one to watch: a large count usually means the pick file and the
waveforms disagree about station naming.
