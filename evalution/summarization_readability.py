from readability import Readability

path = "IAEA_output"
str = ""
with open(path, encoding="utf-8") as f:
    for line in f.readlines():
        str = line
r = Readability(str)
fk = r.flesch_kincaid()
print(fk.score)
print(fk.grade_level)

s = r.smog()
print(s.score)
print(s.grade_level)

dc = r.dale_chall()
print(dc.score)
print(dc.grade_levels)

cl = r.coleman_liau()
print(cl.score)
print(cl.grade_level)

gf = r.gunning_fog()
print(gf.score)
print(gf.grade_level)

# lw = r.linsear_write()
# print(lw.score)
# print(lw.grade_level)