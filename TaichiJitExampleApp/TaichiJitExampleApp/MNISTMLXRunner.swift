//
//  MNISTMLXRunner.swift
//  TaichiJitExampleApp
//
//  Created by Codex on 2026/02/23.
//

import Foundation
import MLX

enum MNISTMLXRunnerError: LocalizedError {
    case missingBundleResource(String)
    case invalidDatasetFormat(String)
    case invalidPrediction(String)

    var errorDescription: String? {
        switch self {
        case .missingBundleResource(let name):
            return "Missing app resource: \(name)"
        case .invalidDatasetFormat(let detail):
            return "Invalid MNIST dataset format: \(detail)"
        case .invalidPrediction(let detail):
            return "Invalid model prediction: \(detail)"
        }
    }
}

final class MNISTMLXRunner {
    private struct MNISTBinaryDataset {
        let imageSize: Int
        let trainCount: Int
        let testCount: Int
        let trainImages: [Float]
        let trainLabels: [UInt8]
        let testImages: [Float]
        let testLabels: [UInt8]

        static func load(from url: URL) throws -> MNISTBinaryDataset {
            let data = try Data(contentsOf: url)
            let headerSize = 4 + 4 + 4 + 4 + 4
            guard data.count >= headerSize else {
                throw MNISTMLXRunnerError.invalidDatasetFormat("file too small")
            }

            var offset = 0
            let magic = String(data: data[offset..<offset + 4], encoding: .ascii) ?? ""
            offset += 4
            guard magic == "MNST" else {
                throw MNISTMLXRunnerError.invalidDatasetFormat("bad magic: \(magic)")
            }

            let version = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt32.self) }
            offset += 4
            guard version == 1 else {
                throw MNISTMLXRunnerError.invalidDatasetFormat("unsupported version: \(version)")
            }

            let imageSizeU32 = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt32.self) }
            offset += 4
            let trainCountU32 = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt32.self) }
            offset += 4
            let testCountU32 = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt32.self) }
            offset += 4

            let imageSize = Int(imageSizeU32)
            let trainCount = Int(trainCountU32)
            let testCount = Int(testCountU32)

            let trainImageBytes = trainCount * imageSize * MemoryLayout<Float>.size
            let trainLabelBytes = trainCount
            let testImageBytes = testCount * imageSize * MemoryLayout<Float>.size
            let testLabelBytes = testCount
            let totalBytes = headerSize + trainImageBytes + trainLabelBytes + testImageBytes + testLabelBytes
            guard data.count == totalBytes else {
                throw MNISTMLXRunnerError.invalidDatasetFormat("byte size mismatch")
            }

            let trainImages: [Float] = data[offset..<offset + trainImageBytes].withUnsafeBytes {
                Array($0.bindMemory(to: Float.self))
            }
            offset += trainImageBytes

            let trainLabels: [UInt8] = Array(data[offset..<offset + trainLabelBytes])
            offset += trainLabelBytes

            let testImages: [Float] = data[offset..<offset + testImageBytes].withUnsafeBytes {
                Array($0.bindMemory(to: Float.self))
            }
            offset += testImageBytes

            let testLabels: [UInt8] = Array(data[offset..<offset + testLabelBytes])

            return MNISTBinaryDataset(
                imageSize: imageSize,
                trainCount: trainCount,
                testCount: testCount,
                trainImages: trainImages,
                trainLabels: trainLabels,
                testImages: testImages,
                testLabels: testLabels
            )
        }
    }

    private let dataset: MNISTBinaryDataset
    private let trainLabels: [Int32]
    private let testLabels: [Int32]
    private let linearWithBias: ([MLXArray]) -> [MLXArray]

    private var weights: MLXArray
    private var bias: MLXArray

    private let numClasses = 10
    private let batchSize = 128

    init() throws {
        let bundle = Bundle.main
        guard let datasetURL = bundle.url(forResource: "mnist_subset", withExtension: "bin") else {
            throw MNISTMLXRunnerError.missingBundleResource("MNIST/mnist_subset.bin")
        }

        let dataset = try MNISTBinaryDataset.load(from: datasetURL)
        guard dataset.imageSize == 28 * 28 else {
            throw MNISTMLXRunnerError.invalidDatasetFormat("unexpected image size: \(dataset.imageSize)")
        }
        self.dataset = dataset

        self.trainLabels = dataset.trainLabels.map(Int32.init)
        self.testLabels = dataset.testLabels.map(Int32.init)

        let weightScale: Float = 0.01
        let initialWeights = (0..<(numClasses * dataset.imageSize)).map { _ in
            Float.random(in: -weightScale...weightScale)
        }
        self.weights = MLXArray(initialWeights, [numClasses, dataset.imageSize])
        self.bias = MLXArray.zeros([numClasses], type: Float.self)

        let forwardKernel = MLXFast.metalKernel(
            name: "mnist_linear_forward",
            inputNames: ["x", "w", "b"],
            outputNames: ["out"],
            source: """
                uint elem = thread_position_in_grid.x;
                int batch = x_shape[0];
                int inFeatures = x_shape[1];
                int outFeatures = w_shape[0];

                int outIndex = int(elem);
                int cls = outIndex % outFeatures;
                int sample = outIndex / outFeatures;
                if (sample >= batch) return;

                float acc = b[cls];
                int xBase = sample * inFeatures;
                int wBase = cls * inFeatures;
                for (int i = 0; i < inFeatures; ++i) {
                    acc += x[xBase + i] * w[wBase + i];
                }
                out[outIndex] = acc;
                """
        )

        let gradXKernel = MLXFast.metalKernel(
            name: "mnist_linear_grad_x",
            inputNames: ["w", "cotangent"],
            outputNames: ["x_grad"],
            source: """
                uint elem = thread_position_in_grid.x;
                int batch = cotangent_shape[0];
                int outFeatures = cotangent_shape[1];
                int inFeatures = w_shape[1];

                int outIndex = int(elem);
                int feature = outIndex % inFeatures;
                int sample = outIndex / inFeatures;
                if (sample >= batch) return;

                float acc = 0.0f;
                int cotBase = sample * outFeatures;
                for (int cls = 0; cls < outFeatures; ++cls) {
                    acc += cotangent[cotBase + cls] * w[cls * inFeatures + feature];
                }
                x_grad[outIndex] = acc;
                """
        )

        let gradWKernel = MLXFast.metalKernel(
            name: "mnist_linear_grad_w",
            inputNames: ["x", "cotangent"],
            outputNames: ["w_grad"],
            source: """
                uint elem = thread_position_in_grid.x;
                int batch = x_shape[0];
                int inFeatures = x_shape[1];
                int outFeatures = cotangent_shape[1];

                int outIndex = int(elem);
                int feature = outIndex % inFeatures;
                int cls = outIndex / inFeatures;
                if (cls >= outFeatures) return;

                float acc = 0.0f;
                for (int sample = 0; sample < batch; ++sample) {
                    acc += cotangent[sample * outFeatures + cls] * x[sample * inFeatures + feature];
                }
                w_grad[outIndex] = acc;
                """
        )

        let gradBKernel = MLXFast.metalKernel(
            name: "mnist_linear_grad_b",
            inputNames: ["cotangent"],
            outputNames: ["b_grad"],
            source: """
                uint elem = thread_position_in_grid.x;
                int batch = cotangent_shape[0];
                int outFeatures = cotangent_shape[1];
                int cls = int(elem);
                if (cls >= outFeatures) return;

                float acc = 0.0f;
                for (int sample = 0; sample < batch; ++sample) {
                    acc += cotangent[sample * outFeatures + cls];
                }
                b_grad[cls] = acc;
                """
        )

        self.linearWithBias = CustomFunction {
            Forward { [forwardKernel] inputs in
                let x = inputs[0]
                let w = inputs[1]
                let b = inputs[2]
                let batch = x.shape[0]
                let outFeatures = w.shape[0]
                let outputShape = [batch, outFeatures]
                let total = batch * outFeatures

                return forwardKernel(
                    [x, w, b],
                    grid: (total, 1, 1),
                    threadGroup: (128, 1, 1),
                    outputShapes: [outputShape],
                    outputDTypes: [x.dtype]
                )
            }

            VJP { [gradXKernel, gradWKernel, gradBKernel] primals, cotangents in
                let x = primals[0]
                let w = primals[1]
                let cotangent = cotangents[0]
                let batch = x.shape[0]
                let inFeatures = x.shape[1]
                let outFeatures = w.shape[0]

                let gradX = gradXKernel(
                    [w, cotangent],
                    grid: (batch * inFeatures, 1, 1),
                    threadGroup: (128, 1, 1),
                    outputShapes: [[batch, inFeatures]],
                    outputDTypes: [x.dtype]
                )[0]

                let gradW = gradWKernel(
                    [x, cotangent],
                    grid: (outFeatures * inFeatures, 1, 1),
                    threadGroup: (128, 1, 1),
                    outputShapes: [[outFeatures, inFeatures]],
                    outputDTypes: [w.dtype]
                )[0]

                let gradB = gradBKernel(
                    [cotangent],
                    grid: (outFeatures, 1, 1),
                    threadGroup: (128, 1, 1),
                    outputShapes: [[outFeatures]],
                    outputDTypes: [w.dtype]
                )[0]

                return [gradX, gradW, gradB]
            }
        }

        eval(weights, bias)
    }

    func train(
        epochs: Int,
        learningRate: Float,
        lossThreshold: Float?,
        progress: ((MNISTTrainingProgress) -> Void)?
    ) throws -> MNISTTrainingSummary {
        let targetEpochs = max(1, epochs)
        let trainCount = dataset.trainCount

        let lossAndGrad = valueAndGrad(
            { [linearWithBias] args in
                let x = args[0]
                let y = args[1]
                let w = args[2]
                let b = args[3]
                let logits = linearWithBias([x, w, b])[0]
                let loss = Self.crossEntropyLoss(logits: logits, targets: y)
                return [loss]
            },
            argumentNumbers: [2, 3]
        )

        var completedEpochs = 0
        var stoppedByLossThreshold = false
        var lastAvgLoss: Float = 0
        var lastAvgAccuracy: Float = 0

        for epoch in 0..<targetEpochs {
            var seen = 0
            var weightedLossSum: Float = 0
            var weightedAccSum: Float = 0
            var nextProgressSample = 100

            while seen < trainCount {
                let count = min(batchSize, trainCount - seen)
                let xBatch = makeFeatureArray(from: dataset.trainImages, start: seen, count: count)
                let yBatch = makeLabelArray(from: trainLabels, start: seen, count: count)

                let (lossOutputs, grads) = lossAndGrad([xBatch, yBatch, weights, bias])
                let batchLoss = lossOutputs[0]

                weights = weights - learningRate * grads[0]
                bias = bias - learningRate * grads[1]

                let logits = linearWithBias([xBatch, weights, bias])[0]
                let predicted = argMax(logits, axis: -1)
                let batchAccuracy = mean((predicted .== yBatch).asType(.float32))

                eval(weights, bias, batchLoss, batchAccuracy)

                let batchLossValue = batchLoss.item(Float.self)
                let batchAccuracyValue = batchAccuracy.item(Float.self)
                weightedLossSum += batchLossValue * Float(count)
                weightedAccSum += batchAccuracyValue * Float(count)
                seen += count

                if seen >= nextProgressSample || seen == trainCount {
                    let avgLoss = weightedLossSum / Float(seen)
                    let avgAccuracy = weightedAccSum / Float(seen)
                    progress?(
                        MNISTTrainingProgress(
                            epoch: epoch + 1,
                            sample: seen,
                            totalSamples: trainCount,
                            averageLoss: avgLoss,
                            accuracy: avgAccuracy
                        )
                    )
                    lastAvgLoss = avgLoss
                    lastAvgAccuracy = avgAccuracy
                    nextProgressSample += 100

                    if let threshold = lossThreshold, avgLoss <= threshold {
                        stoppedByLossThreshold = true
                        break
                    }
                }
            }

            completedEpochs = epoch + 1
            if stoppedByLossThreshold {
                break
            }
        }

        let testAccuracy = try evaluateTestAccuracy(sampleLimit: min(dataset.testCount, 300))
        return MNISTTrainingSummary(
            epochs: completedEpochs,
            trainLoss: lastAvgLoss,
            trainAccuracy: lastAvgAccuracy,
            testAccuracy: testAccuracy,
            stoppedByLossThreshold: stoppedByLossThreshold
        )
    }

    func inferRandomTestSample() throws -> MNISTInferenceResult {
        let index = Int.random(in: 0..<dataset.testCount)
        return try inferTestSample(index: index)
    }

    private func inferTestSample(index: Int) throws -> MNISTInferenceResult {
        let imageStart = index * dataset.imageSize
        let imageEnd = imageStart + dataset.imageSize
        let image = Array(dataset.testImages[imageStart..<imageEnd])

        let x = MLXArray(image, [1, dataset.imageSize])
        let logits = linearWithBias([x, weights, bias])[0]
        let probabilities = softmax(logits, axis: -1)
        let predictedArray = argMax(logits, axis: -1)
        eval(probabilities, predictedArray)

        let predictedLabel = Int(predictedArray.item(Int32.self))
        let confidenceCandidates = probabilities.reshaped([numClasses]).asArray(Float.self)
        guard confidenceCandidates.indices.contains(predictedLabel) else {
            throw MNISTMLXRunnerError.invalidPrediction("predicted label out of range: \(predictedLabel)")
        }

        return MNISTInferenceResult(
            image: image,
            label: Int(dataset.testLabels[index]),
            predictedLabel: predictedLabel,
            confidence: confidenceCandidates[predictedLabel]
        )
    }

    private func evaluateTestAccuracy(sampleLimit: Int) throws -> Float {
        let total = max(1, min(sampleLimit, dataset.testCount))
        var seen = 0
        var correct: Float = 0

        while seen < total {
            let count = min(batchSize, total - seen)
            let xBatch = makeFeatureArray(from: dataset.testImages, start: seen, count: count)
            let yBatch = makeLabelArray(from: testLabels, start: seen, count: count)

            let logits = linearWithBias([xBatch, weights, bias])[0]
            let predicted = argMax(logits, axis: -1)
            let batchCorrect = sum((predicted .== yBatch).asType(.float32))
            eval(batchCorrect)

            correct += batchCorrect.item(Float.self)
            seen += count
        }

        return correct / Float(total)
    }

    private static func crossEntropyLoss(logits: MLXArray, targets: MLXArray) -> MLXArray {
        let selectedLogit = takeAlong(
            logits,
            targets.expandedDimensions(axis: -1),
            axis: -1
        ).squeezed(axis: -1)
        return mean(logSumExp(logits, axis: -1) - selectedLogit)
    }

    private func makeFeatureArray(from imageData: [Float], start: Int, count: Int) -> MLXArray {
        let lower = start * dataset.imageSize
        let upper = lower + count * dataset.imageSize
        return MLXArray(Array(imageData[lower..<upper]), [count, dataset.imageSize])
    }

    private func makeLabelArray(from labels: [Int32], start: Int, count: Int) -> MLXArray {
        MLXArray(Array(labels[start..<start + count]), [count])
    }
}
