#!/usr/bin/env node
// JS executor used by sagetransval.js_adapter.
//
// Reads SAGE_INPUT_JSON path from env, loads the source module, calls the entry
// point with positional args, and prints a JSON envelope to stdout. Errors are
// captured as {"exception_class": "...", "exception_message": "..."} rather than
// crashing the process so the harness can observe them.
//
// Usage:
//   SAGE_SOURCE=source.js SAGE_ENTRYPOINT=fn SAGE_INPUT_JSON=input.json node js_executor.js

const fs = require("fs");
const path = require("path");

function main() {
  const src = process.env.SAGE_SOURCE;
  const entry = process.env.SAGE_ENTRYPOINT;
  const inputPath = process.env.SAGE_INPUT_JSON;
  if (!src || !entry || !inputPath) {
    console.log(JSON.stringify({
      return_value: null,
      exception_class: "ConfigError",
      exception_message: "SAGE_SOURCE / SAGE_ENTRYPOINT / SAGE_INPUT_JSON required",
    }));
    return;
  }

  let mod;
  try {
    mod = require(path.resolve(src));
  } catch (e) {
    console.log(JSON.stringify({
      return_value: null,
      exception_class: e.constructor.name,
      exception_message: String(e.message || e),
      load_error: true,
    }));
    return;
  }

  if (typeof mod[entry] !== "function") {
    console.log(JSON.stringify({
      return_value: null,
      exception_class: "EntryNotFound",
      exception_message: `entrypoint ${entry} is not a function in ${src}`,
    }));
    return;
  }

  let args;
  try {
    args = JSON.parse(fs.readFileSync(inputPath, "utf-8"));
  } catch (e) {
    console.log(JSON.stringify({
      return_value: null,
      exception_class: "InputError",
      exception_message: String(e.message || e),
    }));
    return;
  }

  // args can be either {"__positional__": [...]} or {"name": value, ...}
  let positional;
  if (Object.prototype.hasOwnProperty.call(args, "__positional__")) {
    positional = args["__positional__"];
  } else {
    // Order keys by declaration order from SAGE_PARAM_ORDER if provided
    const order = (process.env.SAGE_PARAM_ORDER || "").split(",").filter(x => x);
    positional = order.length ? order.map(k => args[k]) : Object.values(args);
  }

  try {
    const result = mod[entry].apply(null, positional);
    if (result && typeof result.then === "function") {
      // Async function: not supported in v1 of the harness
      console.log(JSON.stringify({
        return_value: null,
        exception_class: "AsyncNotSupported",
        exception_message: "entrypoint returned a Promise; only sync functions supported",
      }));
      return;
    }
    console.log(JSON.stringify({
      return_value: result,
      exception_class: null,
    }, (_k, v) => {
      if (typeof v === "number" && !Number.isFinite(v)) {
        if (Number.isNaN(v)) return "NaN";
        if (v === Infinity) return "Infinity";
        if (v === -Infinity) return "-Infinity";
      }
      return v;
    }));
  } catch (e) {
    console.log(JSON.stringify({
      return_value: null,
      exception_class: e && e.constructor && e.constructor.name || "Error",
      exception_message: String(e && e.message != null ? e.message : e),
    }));
  }
}

main();
