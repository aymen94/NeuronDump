#!/usr/bin/env node
// Runs neurondump from the OS package npm installed alongside this one (an optionalDependency;
// npm only installs the one matching the host's os and cpu).
"use strict";

const os = { win32: "windows", darwin: "macos", linux: "linux" }[process.platform];

let run;
try {
  run = require.resolve(`@aymen94/neurondump-${os}/bin/neurondump.js`);
} catch {
  console.error(
    `neurondump: no prebuilt binary for ${process.platform}-${process.arch}.\n` +
      "Supported: Windows x64, macOS arm64 / x64, Linux x64 (glibc).\n" +
      "If you installed with --no-optional or --omit=optional, reinstall without it.\n" +
      "Otherwise install with Rust instead: cargo install neurondump"
  );
  process.exit(1);
}
require(run);
