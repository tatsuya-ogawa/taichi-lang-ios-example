//
//  TaichiKernelRunner.swift
//  TaichiJitExampleApp
//
//  Created by Codex on 2026/02/11.
//

import Foundation
import Metal

enum TaichiRunnerError: LocalizedError {
    case metalUnavailable
    case commandQueueUnavailable
    case shaderNotFound(String)
    case pipelineCreationFailed(String)
    case commandBufferFailed(String)

    var errorDescription: String? {
        switch self {
        case .metalUnavailable:
            return "Metal device is unavailable on this device."
        case .commandQueueUnavailable:
            return "Failed to create Metal command queue."
        case .shaderNotFound(let name):
            return "Missing shader resource: \(name).metallib"
        case .pipelineCreationFailed(let name):
            return "Failed to create compute pipeline for \(name)."
        case .commandBufferFailed(let detail):
            return "Compute command failed: \(detail)"
        }
    }
}

struct TaichiRunResult {
    let loss: Float
    let x: [Float]
    let xGrad: [Float]
    let maxAbsGradError: Float

    func pretty() -> String {
        var lines: [String] = []
        lines.append("forward/backward finished")
        lines.append(String(format: "loss = %.6f", loss))
        lines.append(String(format: "max |grad - expected| = %.6f", maxAbsGradError))
        lines.append("")
        lines.append("first 4 entries")
        for i in 0..<min(4, x.count) {
            lines.append(String(format: "i=%d  x=%.6f  grad=%.6f", i, x[i], xGrad[i]))
        }
        return lines.joined(separator: "\n")
    }
}

final class TaichiKernelRunner {
    private struct InitArgs {
        var base: Float
        var pad0: UInt32 = 0
        var pad1: UInt32 = 0
        var pad2: UInt32 = 0
    }

    private static let xCount = 16
    private static let rootBufferBytes = 136
    private static let u32Count = rootBufferBytes / MemoryLayout<UInt32>.size
    private static let lossIndex = 0
    private static let lossGradIndex = 1
    private static let xStartIndex = 2
    private static let xGradStartIndex = 18

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let initPipeline: MTLComputePipelineState
    private let clearLossPipeline: MTLComputePipelineState
    private let forwardPipeline: MTLComputePipelineState
    private let backwardPipeline: MTLComputePipelineState

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TaichiRunnerError.metalUnavailable
        }
        guard let commandQueue = device.makeCommandQueue() else {
            throw TaichiRunnerError.commandQueueUnavailable
        }
        self.device = device
        self.commandQueue = commandQueue
        self.initPipeline = try Self.makePipeline(device: device, libraryName: "init_x")
        self.clearLossPipeline = try Self.makePipeline(device: device, libraryName: "clear_loss")
        self.forwardPipeline = try Self.makePipeline(device: device, libraryName: "forward")
        self.backwardPipeline = try Self.makePipeline(device: device, libraryName: "backward")
    }

    func run(base: Float = 0.25) throws -> TaichiRunResult {
        guard let rootBuffer = device.makeBuffer(length: Self.rootBufferBytes, options: .storageModeShared) else {
            throw TaichiRunnerError.commandBufferFailed("failed to allocate root buffer")
        }
        rootBuffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: Self.rootBufferBytes)

        try encodeInit(base: base, rootBuffer: rootBuffer)
        try encodeNoArgKernel(pipeline: clearLossPipeline, rootBuffer: rootBuffer, threadCount: 1, threadsPerGroup: 1)
        try encodeNoArgKernel(
            pipeline: forwardPipeline,
            rootBuffer: rootBuffer,
            threadCount: Self.xCount,
            threadsPerGroup: 16
        )

        let u32 = rootBuffer.contents().bindMemory(to: UInt32.self, capacity: Self.u32Count)
        u32[Self.lossGradIndex] = Float(1.0).bitPattern
        for i in 0..<Self.xCount {
            u32[Self.xGradStartIndex + i] = 0
        }

        try encodeNoArgKernel(
            pipeline: backwardPipeline,
            rootBuffer: rootBuffer,
            threadCount: Self.xCount,
            threadsPerGroup: 16
        )

        let loss = Float(bitPattern: u32[Self.lossIndex])
        var x: [Float] = []
        var xGrad: [Float] = []
        x.reserveCapacity(Self.xCount)
        xGrad.reserveCapacity(Self.xCount)
        var maxAbsErr: Float = 0
        for i in 0..<Self.xCount {
            let xv = Float(bitPattern: u32[Self.xStartIndex + i])
            let gv = Float(bitPattern: u32[Self.xGradStartIndex + i])
            let expected = 2 * xv
            maxAbsErr = max(maxAbsErr, abs(gv - expected))
            x.append(xv)
            xGrad.append(gv)
        }
        return TaichiRunResult(loss: loss, x: x, xGrad: xGrad, maxAbsGradError: maxAbsErr)
    }

    private static func makePipeline(device: MTLDevice, libraryName: String) throws -> MTLComputePipelineState {
        let library = try makeLibrary(device: device, libraryName: libraryName)
        guard let function = library.makeFunction(name: "main0") else {
            throw TaichiRunnerError.pipelineCreationFailed(libraryName)
        }
        return try device.makeComputePipelineState(function: function)
    }

    private static func makeLibrary(device: MTLDevice, libraryName: String) throws -> MTLLibrary {
        let bundle = Bundle.main
        let direct = bundle.url(forResource: libraryName, withExtension: "metallib")
        let nested = bundle.url(forResource: libraryName, withExtension: "metallib", subdirectory: "Shaders")
        guard let url = direct ?? nested else {
            throw TaichiRunnerError.shaderNotFound(libraryName)
        }
        return try device.makeLibrary(URL: url)
    }

    private func encodeInit(base: Float, rootBuffer: MTLBuffer) throws {
        var args = InitArgs(base: base)
        guard let argsBuffer = device.makeBuffer(bytes: &args, length: MemoryLayout<InitArgs>.stride, options: .storageModeShared) else {
            throw TaichiRunnerError.commandBufferFailed("failed to allocate init args buffer")
        }
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw TaichiRunnerError.commandBufferFailed("failed to create command buffer for init")
        }
        encoder.setComputePipelineState(initPipeline)
        encoder.setBuffer(argsBuffer, offset: 0, index: 0)
        encoder.setBuffer(rootBuffer, offset: 0, index: 1)
        encoder.dispatchThreads(
            MTLSize(width: Self.xCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 16, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw TaichiRunnerError.commandBufferFailed(error.localizedDescription)
        }
    }

    private func encodeNoArgKernel(
        pipeline: MTLComputePipelineState,
        rootBuffer: MTLBuffer,
        threadCount: Int,
        threadsPerGroup: Int
    ) throws {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw TaichiRunnerError.commandBufferFailed("failed to create command buffer")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(rootBuffer, offset: 0, index: 0)
        encoder.dispatchThreads(
            MTLSize(width: threadCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw TaichiRunnerError.commandBufferFailed(error.localizedDescription)
        }
    }
}

