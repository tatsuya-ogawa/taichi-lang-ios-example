// Minimal Slang autodiff probe for Metal backend verification.
// It checks both default reverse-mode and custom backward registration.

RWStructuredBuffer<float> inValue;
RWStructuredBuffer<float> outGradAuto;
RWStructuredBuffer<float> outGradCustom;

[Differentiable]
float squareAuto(float x)
{
    return x * x;
}

void squareCustomBackward(inout DifferentialPair<float> x, float dOut)
{
    x = DifferentialPair<float>(x.p, x.d + (2.0f * x.p) * dOut);
}

[BackwardDerivative(squareCustomBackward)]
float squareCustom(float x)
{
    return x * x;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void run_backward_auto(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    DifferentialPair<float> x = diffPair(inValue[i]);
    bwd_diff(squareAuto)(x, 1.0f);
    outGradAuto[i] = x.d;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void run_backward_custom(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    DifferentialPair<float> x = diffPair(inValue[i]);
    bwd_diff(squareCustom)(x, 1.0f);
    outGradCustom[i] = x.d;
}
