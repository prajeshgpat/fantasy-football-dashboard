import io, contextlib, runpy
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    g = runpy.run_path(str(__import__("pathlib").Path(__file__).with_name("league.py")))
df, best_lineup = g["df"], g["best_lineup"]
ME = "Cotton Picken Higgers (2-1) ME"

def total(team, col="ros_ppg"):
    return best_lineup(df[df.team_owner == team], col)[col].sum()

def owner_of(name):
    return df[df.name.str.contains(name, regex=False)].iloc[0].team_owner

def trade(give, get, col="ros_ppg"):
    other = owner_of(get[0])
    b_me, b_ot = total(ME, col), total(other, col)
    d = df.copy()
    for n in give: d.loc[d.name.str.contains(n, regex=False), "team_owner"] = other
    for n in get:  d.loc[d.name.str.contains(n, regex=False), "team_owner"] = ME
    def tot(t):
        return best_lineup(d[d.team_owner == t], col)[col].sum()
    print(f"{' + '.join(give):38s} <-> {' + '.join(get):26s} [{other[:18]:18s}] me {tot(ME)-b_me:+5.1f}  them {tot(other)-b_ot:+5.1f}")

for col in ["ros_ppg"]:
    trade(["Saquon Barkley", "Stefon Diggs"], ["Amon-Ra St. Brown"])
    trade(["Jeremiyah Love", "Stefon Diggs"], ["Amon-Ra St. Brown"])
    trade(["Saquon Barkley", "Jeremiyah Love"], ["Amon-Ra St. Brown"])
    trade(["Saquon Barkley", "Jalen Coker"], ["Amon-Ra St. Brown"])
    trade(["Stefon Diggs", "Isaiah Likely"], ["Lamar Jackson"])
    trade(["Stefon Diggs"], ["Lamar Jackson"])
    trade(["Joe Burrow", "Stefon Diggs"], ["Lamar Jackson", "DeVonta Smith"])
    trade(["Saquon Barkley", "Stefon Diggs"], ["Ja'Marr Chase"])
    trade(["Jeremiyah Love", "Joe Burrow"], ["Ja'Marr Chase"])
    trade(["Saquon Barkley", "Stefon Diggs"], ["Puka Nacua"])
    trade(["Jeremiyah Love", "Jalen Coker"], ["Puka Nacua"])
    trade(["Stefon Diggs", "Isaiah Likely"], ["Dak Prescott", "George Kittle"])
    trade(["Jeremiyah Love", "Stefon Diggs"], ["Justin Jefferson"])
    trade(["Saquon Barkley", "Stefon Diggs"], ["Rashee Rice"])
    trade(["Saquon Barkley", "Stefon Diggs"], ["Christian McCaffrey"])
    trade(["Jeremiyah Love", "Jalen Coker"], ["Jahmyr Gibbs"])
