#!/usr/bin/env bash
# Download the Delta Lake jars for Spark 3.5 / Scala 2.12 into spark/jars/ (once; gitignored).
# Mounted into both Spark containers at /opt/spark/extra-jars and passed with --jars, so
# spark-submit needs no --packages, no network and no ivy cache at run time.
#   delta-spark_2.12-3.2.1.jar  (Delta 3.2.x is built against Spark 3.5)
#   delta-storage-3.2.1.jar     (its only runtime dependency not shipped with Spark;
#                                antlr4-runtime 4.9.3 already ships with Spark 3.5)
# SHA-1 values from Maven Central (<jar>.sha1).
set -eu
DIR="$(cd "$(dirname "$0")" && pwd)/jars"
BASE="https://repo1.maven.org/maven2/io/delta"
mkdir -p "$DIR"
fetch() {  # name url sha1
  local f="$DIR/$1"
  if [ -f "$f" ] && [ "$(sha1sum "$f" | cut -d' ' -f1)" = "$3" ]; then echo "ok   $1"; return 0; fi
  curl -fsSL -o "$f.part" "$2"
  if [ "$(sha1sum "$f.part" | cut -d' ' -f1)" != "$3" ]; then rm -f "$f.part"; echo "SHA-1 MISMATCH $1" >&2; return 1; fi
  mv "$f.part" "$f"; echo "got  $1"
}
fetch delta-spark_2.12-3.2.1.jar "$BASE/delta-spark_2.12/3.2.1/delta-spark_2.12-3.2.1.jar" ce09ca3d701f55357199bf5167b3c7daa729a07c
fetch delta-storage-3.2.1.jar    "$BASE/delta-storage/3.2.1/delta-storage-3.2.1.jar"       9d9d56fcae37f1b3d48d80f8b7eefabd3477569d
