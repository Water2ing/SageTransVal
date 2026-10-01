#!/usr/bin/env node
// TypeScript executor: compiles a .ts file to .js with a vendored or installed
// `typescript` package (or falls back to a no-op compile if the file already
// is JS-valid), then re-invokes js_executor with the compiled artifact.
//
// This pre-compile + re-exec is simpler than running tsc as a child and avoids
// pulling in ts-node as a dependency. The translator is expected to emit
// TypeScript that compiles under default strict-mode settings.

const fs = require("fs");
const path = require("path");
const { execSync, spawnSync } = require("child_process");

function findTsc() {
  // Look in node_modules/.bin/tsc relative to a few likely paths.
  const candidates = [
    path.resolve(__dirname, "..", "node_modules", ".bin", "tsc"),
    path.resolve(__dirname, "..", ".venv-node", "node_modules", ".bin", "tsc"),
    "tsc",
  ];
  for (const c of candidates) {
    try {
      execSync(`${c} --version`, { stdio: "ignore" });
      return c;
    } catch (_) {}
  }
  return null;
}

function main() {
  const src = process.env.SAGE_SOURCE;
  if (!src) {
    process.stdout.write(JSON.stringify({
      return_value: null,
      exception_class: "ConfigError",
      exception_message: "SAGE_SOURCE missing",
    }) + "\n");
    return;
  }
  const tsc = findTsc();
  if (!tsc) {
    process.stdout.write(JSON.stringify({
      return_value: null,
      exception_class: "ConfigError",
      exception_message: "tsc not available; run `npm install -D typescript` in the repo root",
    }) + "\n");
    return;
  }
  const outDir = fs.mkdtempSync("/tmp/sage-ts-");
  const compileResult = spawnSync(tsc, [src, "--target", "es2022", "--module", "commonjs", "--outDir", outDir, "--esModuleInterop"], {encoding: "utf-8"});
  if (compileResult.status !== 0) {
    process.stdout.write(JSON.stringify({
      return_value: null,
      exception_class: "CompileError",
      exception_message: (compileResult.stderr || compileResult.stdout || "").trim(),
    }) + "\n");
    return;
  }
  const compiledName = path.basename(src).replace(/\.ts$/, ".js");
  const compiled = path.join(outDir, compiledName);
  if (!fs.existsSync(compiled)) {
    process.stdout.write(JSON.stringify({
      return_value: null,
      exception_class: "CompileError",
      exception_message: `tsc produced no output for ${compiledName}`,
    }) + "\n");
    return;
  }
  // Hand off to the JS executor by replacing SAGE_SOURCE.
  const jsExec = path.resolve(__dirname, "js_executor.js");
  const child = spawnSync("node", [jsExec], {
    env: {...process.env, SAGE_SOURCE: compiled},
    encoding: "utf-8",
  });
  process.stdout.write(child.stdout || "");
  if (child.stderr) process.stderr.write(child.stderr);
  if (child.status !== 0) process.exit(child.status);
}

main();
