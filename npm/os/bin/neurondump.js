#!/usr/bin/env node
// Runs this package's neurondump binary with the same arguments, stdio and exit code.
"use strict";

const { spawnSync } = require("child_process");

const bin = require("..");
const result = spawnSync(bin, process.argv.slice(2), { stdio: "inherit" });
if (result.error) {
  console.error(`neurondump: failed to run ${bin}: ${result.error.message}`);
  process.exit(1);
}
if (result.signal) {
  process.kill(process.pid, result.signal);
}
process.exit(result.status ?? 1);
