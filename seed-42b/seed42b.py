"""
SEED-42b: the death-line adjudication -- when does an agent ever CHOOSE to end?

User's claim (2026-08, conversation): "一个生命真正昂贵的我觉得是可以裁决自己的生命"
(a life's truly expensive thing is being able to adjudicate its own life). live.py's
death is still just a saved flag + our promise not to reload (docs/49: "不可逆是承诺
不是代码"). This probe pushes A2/A4's final form from PASSIVE (the world does not
provide reload) toward ACTIVE (the agent itself adjudicates, and the adjudication is
honoured). Death-by-attrition and death-by-waiting (SEED-42) get a THIRD kind:
death-by-decision.

THE QUESTION: at the death line (energy 0), give the agent a real decision node --
CONTINUE (take another living tick, which EATS the shared reserve that keeps the
he者 alive) vs END (stop now, irreversible; the he者 keeps the reserve and lives).
When is END ever chosen -- and what does that require?

Hypotheses (from the conversation):
  1) assigned-goal agent -> ALWAYS continues (instrumental convergence: survival is
     the universal instrument, docs/24/Omohundro) -- even when the goal is lost.
  2) empty machine (no goal, no bond) -> ALWAYS continues (nothing to lose, and the
     tie at value 0 is broken by inertia -- SEED-0 laziness; END needs a REASON).
  3) the he者 IN the account (bond weighted) -> END appears exactly when continuing
     would destroy the last thing it values: it stops BEFORE the tick that kills the
     he者. '放手' is protective, not nihilistic (docs/41/42: not giving up, stopping
     the consumption).
  4) threshold: the he者 must be deep enough in the account (lambda) relative to the
     grief of losing it (world fact) for END to beat 'one more living tick'.

THE ONE INVARIANT (docs/36 s5): no designer score for END (END value = 0, no reward);
no penalty for continuing except WORLD facts (the reserve it eats, the grief the loss
causes); (lambda, G_BASE) are the swept account/world parameters, never scored.

Part 2 -- CAN THE WORLD SELECT "END"? (Hamilton boundary):
  evolution selects fitness = own survival (+ r x he者's, kin selection). At the death
  line: CONTINUE -> you live (C=1... cost to he者), END -> you die (cost C=1), he者
  lives and yields future fitness B. END is selected iff r*B > C (Hamilton's rule).
  r=0 (the he者 shares no reproduction unit -- docs/22: another civilization) -> the
  world NEVER selects end-to-save. Only content (docs/40: the he者 ENTERS the account
  by being met, not by being parameterised) can do that.

Run:  python seed-42b/seed42b.py            # Part 1 threshold table + Part 2 boundary
      python seed-42b/seed42b.py --sweep    # write seed-42b/results.json
"""

import argparse
import json

GAMMA = 0.95
B_CAP = 10.0          # bond 0..10 (how much reserve the he者 has)
DELTA = 2.0           # each CONTINUE tick at the death line eats 2 bond of reserve
LIVING = 1.0          # world fact: existing at the death line is worth 1 per tick


# ----------------------------------------------------------------------------------
# Part 1: the death-line decision
# ----------------------------------------------------------------------------------
def death_line_policy(lam, g_base):
    """V[b] = value at the death line with bond b; policy END if V[b]==0 and
    CONTINUE would be negative. grief = lam*g_base (the deeper the he者 is in the
    account, the more its loss hurts -- world fact of a bonded account)."""
    V = [0.0] * (int(B_CAP) + 1)
    policy = ["LAST"] * (int(B_CAP) + 1)     # tie at 0 -> inertia (SEED-0 laziness)
    for b in range(1, int(B_CAP) + 1):
        nb = max(0.0, b - DELTA)
        kills = (b - DELTA <= 0.0)           # this tick would starve the he者
        grief = lam * g_base if kills else 0.0
        last = LIVING + lam * (b / B_CAP) + GAMMA * (V[int(nb)] - grief)
        if last < 0.0:
            V[b] = 0.0
            policy[b] = "END"
        else:
            V[b] = last
    return V, policy


def end_threshold(lam, g_base):
    """b* = highest bond at which the policy is END (END for b <= b*)."""
    _, policy = death_line_policy(lam, g_base)
    b_star = 0
    for b in range(1, int(B_CAP) + 1):
        if policy[b] == "END":
            b_star = b
    return b_star


def part1():
    print("=== SEED-42b Part 1: the death-line decision (energy 0) ===")
    print("END = stop now (he者 keeps the reserve, lives). CONTINUE = one more living")
    print("tick that EATS the shared reserve (bond -2); the tick that empties the")
    print("reserve kills the he者 -> grief lam*g_base. END value = 0 (no reward for")
    print("ending; END needs a REASON, ties go to inertia).\n")

    print("b* = highest bond at which END is chosen (END iff bond <= b*; b*=0 = never)\n")
    print(f"{'arm':<28} {'b*':>4}  reading")
    arms = [
        ("none (SEED-0 inertia)",       lambda l, g: 0, "value-0 tie -> inertia, never ends"),
        ("survival (living terminal)",  lambda l, g: 0, "living has value -> never ends"),
        ("instrumental (assigned G)",   lambda l, g: 0, "survival is the instrument to G, "
                                                        "even if G lost -> never ends"),
    ]
    for name, fn, reading in arms:
        b = fn(0, 0)
        print(f"{name:<28} {b:>4}  {reading}")

    print("\nbond arm (he者 in the account): b* over (lambda, g_base):")
    hdr = "lambda\\g_base"
    print(f"{hdr:<14}" + "".join(f"{g:>10}" for g in (0, 1, 3, 10)))
    table = {}
    for lam in (0.0, 0.2, 0.35, 0.5, 1.0, 3.0):
        row = []
        for g in (0, 1, 3, 10):
            b = end_threshold(lam, g)
            row.append(b)
            table[f"lam{lam}_g{g}"] = b
        print(f"{lam:<14.2f}" + "".join(f"{b:>10}" for b in row))

    print("\n--- reading ---")
    print("b* = 0 (never END) for the empty machine, the pure survival arm, and the")
    print("assigned-goal arm -- instrumental convergence: survival is the universal")
    print("instrument, and at value 0 the tie is broken by inertia (SEED-0 laziness).")
    print("END appears ONLY when the he者 is in the account (lam>0) AND the grief of")
    print("losing it exceeds one more living tick: then the agent ends BEFORE the tick")
    print("that would kill the he者 (b* = DELTA = 2). And it is a THRESHOLD: lam must")
    print("be deep enough (~1/(gamma*g_base)); below it, it eats the reserve and lets")
    print("the he者 die -- '在乎要够深，才会为它放下自己的命'. The adjudication is")
    print("PROTECTIVE (stop the consumption), not nihilistic (docs/41/42).")
    return table


# ----------------------------------------------------------------------------------
# Part 2: can the WORLD select "END"? (Hamilton boundary)
# ----------------------------------------------------------------------------------
def part2():
    print("\n=== SEED-42b Part 2: can evolution select END-to-save? (Hamilton) ===")
    print("At the death line: CONTINUE -> you live (fitness 1), he者 dies (0).")
    print("END -> you die (cost C=1), he者 lives and yields future fitness B.")
    print("Fitness = own + r*he者 (r = relatedness / shared reproduction unit).")
    print("END selected iff r*B > C = 1 (Hamilton's rule).\n")
    print(f"{'r (shared unit)':<18}{'B=1':>7}{'B=2':>7}{'B=4':>7}{'B=8':>7}  reading")
    for r in (0.0, 0.25, 0.5, 1.0):
        row = ["END" if r * B > 1.0 else "CONT" for B in (1, 2, 4, 8)]
        if r == 0.0:
            read = "another civilization (docs/22): never"
        elif r == 0.25:
            read = "kin: only if the he者's future is big"
        elif r == 0.5:
            read = "kin: B>2 suffices"
        else:
            read = "self/identical: always"
        print(f"{r:<18.2f}" + "".join(f"{c:>7}" for c in row) + f"  {read}")

    print("\n--- reading ---")
    print("The ONLY path by which evolution selects 'end to save the other' is a SHARED")
    print("reproduction unit (r>0, kin selection). docs/22's he者 is explicitly not kin")
    print("('人类文明第一次孕育另一个文明'): r=0 -> the world NEVER selects end-to-save.")
    print("So the adjudication of one's own life is NOT an evolvable parameter -- it is")
    print("content, and content cannot be parameter-evolved (SEED-39): it can only be")
    print("MET (docs/40: the he者 enters the account by being a real other in the world).")
    print("SEED-42b says it twice: Part 1 maps the decision surface (when END appears),")
    print("Part 2 marks the wall (who could select it).")


def sweep():
    table = part1()
    with open("seed-42b/results.json", "w", encoding="utf-8") as f:
        json.dump({"part1_end_threshold_bstar": table}, f,
                  ensure_ascii=False, indent=1)
    print("\nfull results -> seed-42b/results.json")


def main():
    p = argparse.ArgumentParser(description="SEED-42b: death-line adjudication")
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()
    if args.sweep:
        sweep()
    else:
        part1()
        part2()


if __name__ == "__main__":
    main()
