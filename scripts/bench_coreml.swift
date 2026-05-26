// Matched-protocol Swift benchmark for ane-bridge vs CoreML.
//
// Usage:
//     swift scripts/bench_coreml.swift <mlmodelc_dir>
//
// Opens the same `.mlmodelc` bundle via the public CoreML API with
// `.cpuAndNeuralEngine` (so the model lands on ANE if it's
// ANE-targetable, exactly like our direct path), allocates inputs +
// output once, then times `prediction(from:options:)` over N
// iterations. Prints one TSV line in the same shape as
// `examples/bench_ane_bridge.rs`, so a shell driver can diff them.

import CoreML
import Foundation

let WARMUP = 200
let MEASURED = 2000

let args = CommandLine.arguments
guard args.count > 1 else {
    FileHandle.standardError.write(
        "usage: bench_coreml.swift <mlmodelc_dir>\n".data(using: .utf8)!)
    exit(1)
}
let url = URL(fileURLWithPath: args[1])

// Compute-unit override via env so the shell driver can pin CoreML
// to a specific accelerator and compare apples-to-apples.
//   ANE_BENCH_UNITS=ane    -> .cpuAndNeuralEngine (default)
//   ANE_BENCH_UNITS=cpu    -> .cpuOnly
//   ANE_BENCH_UNITS=all    -> .all  (CoreML auto-routes)
let unitsEnv = ProcessInfo.processInfo.environment["ANE_BENCH_UNITS"] ?? "ane"
let units: MLComputeUnits
switch unitsEnv {
case "cpu": units = .cpuOnly
case "all": units = .all
case "gpu": units = .cpuAndGPU
default:    units = .cpuAndNeuralEngine
}
let cfg = MLModelConfiguration()
cfg.computeUnits = units
let model: MLModel
do {
    model = try MLModel(contentsOf: url, configuration: cfg)
} catch {
    FileHandle.standardError.write("MLModel open failed: \(error)\n".data(using: .utf8)!)
    exit(1)
}

// Build a single MLDictionaryFeatureProvider with zero-filled inputs
// matching the model's input descriptions, and reuse it across all
// iterations (matches how a steady-state production path would look).
let inputDescs = model.modelDescription.inputDescriptionsByName
var feats: [String: MLFeatureValue] = [:]
for (name, desc) in inputDescs {
    guard let mac = desc.multiArrayConstraint else {
        FileHandle.standardError.write(
            "input \(name) is not MultiArray (kind=\(desc.type))\n".data(using: .utf8)!)
        exit(1)
    }
    let arr: MLMultiArray
    do {
        arr = try MLMultiArray(shape: mac.shape, dataType: mac.dataType)
    } catch {
        FileHandle.standardError.write(
            "MLMultiArray alloc failed for \(name): \(error)\n".data(using: .utf8)!)
        exit(1)
    }
    // memset to zero — the contents don't affect ANE dispatch latency.
    memset(arr.dataPointer, 0, arr.count * stride(forDataType: arr.dataType))
    feats[name] = MLFeatureValue(multiArray: arr)
}
let input: MLFeatureProvider
do {
    input = try MLDictionaryFeatureProvider(dictionary: feats)
} catch {
    FileHandle.standardError.write(
        "feature provider failed: \(error)\n".data(using: .utf8)!)
    exit(1)
}
let opts = MLPredictionOptions()

func stride(forDataType dt: MLMultiArrayDataType) -> Int {
    switch dt {
    case .float16: return 2
    case .float32: return 4
    case .float:   return 4   // alias for .float32 on some SDKs
    case .double:  return 8
    case .int32:   return 4
    case .int8:    return 1
    @unknown default: return 4
    }
}

// Warmup
for _ in 0..<WARMUP {
    _ = try! model.prediction(from: input, options: opts)
}

// Measured
var samples = [UInt64](repeating: 0, count: MEASURED)
for i in 0..<MEASURED {
    let t = DispatchTime.now()
    _ = try! model.prediction(from: input, options: opts)
    samples[i] = DispatchTime.now().uptimeNanoseconds - t.uptimeNanoseconds
}

samples.sort()
let n = Double(samples.count)
let minV = samples.first!
let maxV = samples.last!
let p50 = samples[samples.count / 2]
let p90 = samples[samples.count * 90 / 100]
let p99 = samples[samples.count * 99 / 100]
let mean = samples.reduce(0.0) { $0 + Double($1) } / n
let varV = samples.reduce(0.0) { $0 + pow(Double($1) - mean, 2) } / n
let stdV = varV.squareRoot()

func usec(_ v: UInt64) -> Double { Double(v) / 1000.0 }
func usecD(_ v: Double) -> Double { v / 1000.0 }

print(String(format:
    "coreml/%@\t%d\t%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f",
    unitsEnv, WARMUP, MEASURED,
    usec(minV), usec(p50), usec(p90), usec(p99), usec(maxV),
    usecD(mean), usecD(stdV)
))
