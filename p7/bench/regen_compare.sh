#!/usr/bin/env bash
# regen_compare.sh — régénère la comparaison A/B depuis A.tsv/B.tsv existants (sans re-run)
set -uo pipefail
cd /home/guile/llama-experiments/p7/bench
R=${1:-res-20261008-100240}

summary(){ # tsv -> "tok ms tps acc mean" (NA toléré dans acc/mean)
  python3 - "$1" <<'PY'
import sys
rows=[l.split() for l in open(sys.argv[1]) if l.strip()]
te=sum(int(r[3]) for r in rows); tm=sum(float(r[4]) for r in rows)
tps=f"{te/(tm/1e3):.2f}" if tm else "0.00"
accs=[float(r[6]) for r in rows if r[6]!="NA"]
means=[float(r[7]) for r in rows if r[7]!="NA"]
acc=f"{sum(accs)/len(accs):.4f}" if accs else "NA"
mean=f"{sum(means)/len(means):.2f}" if means else "NA"
print(f"{te} {tm:.0f} {tps} {acc} {mean}")
PY
}

SA=$(summary "$R/A.tsv"); SB=$(summary "$R/B.tsv")
echo "SUM_A=$SA"
echo "SUM_B=$SB"
read -r TOKA MS_A TPS_A ACC_A MEAN_A <<< "$SA"
read -r TOKB MS_B TPS_B ACC_B MEAN_B <<< "$SB"
{
echo "================================================================"
echo " RESULTAT A/B — kernel Q6_K P7 (X4) vs PROD (W4R4) — applicatif"
echo " (régénéré depuis $R/A.tsv + $R/B.tsv, sans re-run)"
echo "================================================================"
echo "  A = prod e4afbb4a (W4R4)"
echo "  B = P7   a1b3b776 (X4)"
echo "  taches : T1=256 / T2=512 / T3=256 tok, T=0, MTP actif"
echo "----------------------------------------------------------------"
printf "  %-22s %12s %12s   delta\n" "metrique" "A (prod)" "B (P7)"
printf "  %-22s %12s %12s\n" "eval tok total" "$TOKA" "$TOKB"
printf "  %-22s %12s %12s\n" "eval ms total" "$MS_A" "$MS_B"
python3 - "$TPS_A" "$TPS_B" <<'PY'
import sys; a,b=float(sys.argv[1]),float(sys.argv[2]); print(f"  {'eval t/s (global)':<22} {a:12.2f} {b:12.2f}   {((b-a)/a*100):+.1f}%")
PY
if [ "$ACC_A" != NA ] && [ "$ACC_B" != NA ]; then
  python3 - "$ACC_A" "$ACC_B" <<'PY'
import sys; a,b=float(sys.argv[1]),float(sys.argv[2]); print(f"  {'acceptance moyenne':<22} {a:12.4f} {b:12.4f}   {((b-a)/a*100):+.1f}%")
PY
else echo "  acceptance : NA (A=$ACC_A B=$ACC_B)"; fi
if [ "$MEAN_A" != NA ] && [ "$MEAN_B" != NA ]; then
  python3 - "$MEAN_A" "$MEAN_B" <<'PY'
import sys; a,b=float(sys.argv[1]),float(sys.argv[2]); print(f"  {'mean len moyenne':<22} {a:12.2f} {b:12.2f}   {((b-a)/a*100):+.1f}%")
PY
else echo "  mean len : NA (A=$MEAN_A B=$MEAN_B) — le grep journal ne matche plus la ligne draft acceptance"; fi
echo "================================================================"
} | tee -a "$R/console.log"
