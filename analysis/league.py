import re, sys, itertools, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import pandas as pd
from src import rankings, defense, schedule

TEAMS = {
 "Jalen Its Too Big (3-0)": ["Jalen Hurts","Christian McCaffrey","Cam Skattebo","CeeDee Lamb","Garrett Wilson","Trey McBride","D'Andre Swift","Luther Burden III","Nico Collins","Mike Evans","Tucker Kraft","J.K. Dobbins","Jacory Croskey-Merritt","Caleb Williams"],
 "To Infinity And Bijan (2-1)": ["Trevor Lawrence","Bijan Robinson","Javonte Williams","Justin Jefferson","Drake London","Tyler Warren","Zay Flowers","Parker Washington","Omarion Hampton","Rashee Rice","Jared Goff","Josh Downs","Breece Hall"],
 "Ritz Blitz (2-1)": ["Brock Purdy","Jahmyr Gibbs","Aaron Jones","Deebo Samuel","Devaughn Vele","Dalton Schultz","DeVonta Smith","Travis Etienne","Lamar Jackson","Mark Andrews","Xavier Worthy","Jake Ferguson","Denzel Boston","Emanuel Wilson","A.J. Brown"],
 "Cotton Picken Higgers (2-1) ME": ["Joe Burrow","Jonathan Taylor","Saquon Barkley","George Pickens","Davante Adams","Brock Bowers","Jeremiyah Love","James Cook","Tee Higgins","Stefon Diggs","RJ Harvey","Isaiah Likely","Jalen Coker","Jordan Mason"],
 "Comeback Season (1-2)": ["Drake Maye","Derrick Henry","Kyren Williams","Malik Nabers","Christian Watson","Travis Kelce","George Kittle","Jaylen Warren","Puka Nacua","Bhayshul Tuten","DK Metcalf","Dak Prescott","Kenny Gainwell","Makai Lemon"],
 "Brown Pitts (1-2)": ["Josh Allen","De'Von Achane","Quinshon Judkins","Amon-Ra St. Brown","Chris Olave","Harold Fannin","Tetairoa McMillan","Michael Wilson","Josh Jacobs","Kyle Pitts","Courtland Sutton","Tony Pollard","Kyle Monangai","Jordan Love"],
 "Premature EJEANTYlation (1-2)": ["Justin Herbert","Chase Brown","Ashton Jeanty","Ja'Marr Chase","Ladd McConkey","Dalton Kincaid","Kenneth Walker","Jaylen Waddle","Colston Loveland","Jadarian Price","Rhamondre Stevenson","Brian Thomas","Bo Nix","Rico Dowdle"],
 "My God That's JSN (0-3)": ["Patrick Mahomes","David Montgomery","Bucky Irving","Jaxon Smith-Njigba","Emeka Egbuka","Sam LaPorta","TreVeyon Henderson","Chuba Hubbard","Terry McLaurin","MarShawn Lloyd","DJ Moore","Bryce Young","Matthew Golden","Dallas Goedert"],
}

def norm(s):
    s = s.lower().replace(".", "").replace("'", "").replace("-", " ")
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()

proj, gaps = rankings.project_players()
proj["k"] = proj["full_name"].map(norm)
allp = pd.concat([proj, gaps.assign(k=gaps["full_name"].map(norm))]) if len(gaps) else proj
dvp = defense.defense_vs_position()
summ = pd.DataFrame(schedule.sos(dvp, [])["summary"])
ros = summ.set_index(["team", "position"])["ros"].to_dict()
po = summ.set_index(["team", "position"])["playoffs"].to_dict()

proj["ros_mult"] = [ros.get((t, p)) for t, p in zip(proj["team"], proj["position"])]
proj["po_mult"] = [po.get((t, p)) for t, p in zip(proj["team"], proj["position"])]
proj["ros_ppg"] = proj["skill_ppg"] * proj["ros_mult"].fillna(1.0)
proj["rank_ov"] = proj["skill_ppg"].rank(ascending=False, method="min").astype(int)

owner = {}
rows = []
for team, names in TEAMS.items():
    for n in names:
        m = proj[proj["k"] == norm(n)]
        if m.empty:
            rows.append(dict(team_owner=team, name=n, missing=True))
            continue
        r = m.iloc[0]
        owner[r["k"]] = team
        rows.append(dict(team_owner=team, name=r["full_name"], pos=r["position"], nfl=r["team"], games=int(r["games"]),
                         ppg=r["ppg"], ppg25=r["ppg_2025"], skill=r["skill_ppg"], xppg=r["xppg"],
                         ros=r["ros_mult"], po=r["po_mult"], ros_ppg=r["ros_ppg"], rk=r["rank_ov"]))
df = pd.DataFrame(rows)
print("MISSING:", df[df.get("missing") == True]["name"].tolist() if "missing" in df else [])
df = df[df.get("missing") != True] if "missing" in df else df

def best_lineup(g, col):
    g = g.sort_values(col, ascending=False)
    take = lambda pos, n: g[g["pos"] == pos].head(n)
    qb, rb, wr, te = take("QB", 1), take("RB", 2), take("WR", 2), take("TE", 1)
    used = pd.concat([qb, rb, wr, te])
    flex = g[g["pos"].isin(["RB", "WR", "TE"]) & ~g["name"].isin(used["name"])].head(2)
    return pd.concat([used, flex])

out = []
for team, g in df.groupby("team_owner"):
    lu = best_lineup(g, "skill"); lu2 = best_lineup(g, "ros_ppg")
    bench = g[~g["name"].isin(lu["name"])]
    out.append(dict(team=team, lineup=lu["skill"].sum(), lineup_ros=lu2["ros_ppg"].sum(),
                    QB=lu[lu.pos=="QB"].skill.sum(), RB=lu[lu.pos=="RB"].skill.sum(), WR=lu[lu.pos=="WR"].skill.sum(),
                    TE=lu[lu.pos=="TE"].skill.sum(),
                    flex=lu.groupby("pos").skill.count().to_dict(),
                    bench_top3=bench.skill.nlargest(3).sum()))
res = pd.DataFrame(out).sort_values("lineup_ros", ascending=False)
pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60)
print(res.round(1).to_string(index=False))
print()
for pos in ["QB", "RB", "WR", "TE"]:
    print(pos, "league starter avg skill:", round(df[df.pos == pos].groupby("team_owner").skill.apply(lambda s: s.nlargest({"QB":1,"RB":2,"WR":2,"TE":1}[pos]).mean()).mean(), 1))
print()
cols = ["team_owner","name","pos","nfl","games","ppg","xppg","ppg25","skill","ros","po","ros_ppg","rk"]
print(df.sort_values(["team_owner","skill"], ascending=[True, False])[cols].round(2).to_string(index=False))
print()
# free agents / top unowned
own = set(owner)
fa = proj[~proj["k"].isin(own)].head(20)[["full_name","position","team","games","skill_ppg","xppg","matchup_mult","ros_mult"]]
print("TOP UNOWNED\n", fa.round(2).to_string(index=False))
