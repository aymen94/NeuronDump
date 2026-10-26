// The path of the neurondump binary in this package. The macOS package ships both
// architectures (bin/arm64, bin/x64); the others have one binary in bin/.
"use strict";

const path = require("path");

const exe = process.platform === "win32" ? "neurondump.exe" : "neurondump";
const dir = process.platform === "darwin" ? path.join(__dirname, "bin", process.arch) : path.join(__dirname, "bin");

module.exports = path.join(dir, exe);
