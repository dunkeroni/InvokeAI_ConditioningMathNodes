from typing import Literal, Protocol

import numpy as np
import sympy
import torch

from invokeai.app.invocations.fields import (
    FluxConditioningField,
    CogView4ConditioningField,
    FluxReduxConditioningField,
    TensorField,
)
from invokeai.app.invocations.flux_redux import FluxReduxOutput
from invokeai.app.invocations.primitives import (
    FluxConditioningOutput,
    CogView4ConditioningOutput,
)
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import FLUXConditioningInfo, \
    SD3ConditioningInfo, CogView4ConditioningInfo
from invokeai.invocation_api import (
    BaseInvocation,
    BaseInvocationOutput,
    BasicConditioningInfo,
    ConditioningField,
    ConditioningFieldData,
    ConditioningOutput,
    FieldDescriptions,
    Input,
    InputField,
    InvocationContext,
    OutputField,
    SDXLConditioningInfo,
    invocation,
    invocation_output,
)
from . import torch_funcs

CONDITIONING_OPERATIONS = Literal[
    "LERP",
    "ADD",
    "SUB",
    "APPEND",
    "PERP",
    "PROJ",
]


CONDITIONING_OPERATIONS_LABELS = {
    "LERP": "Linear Interpolation A->B",
    "ADD": "Add A+αB",
    "SUB": "Subtract A-αB",
    "APPEND": "Append [A, B]",
    "PERP": "Perpendicular A⊥B",
    "PROJ": "Projection A||B",
}


def apply_operation(operation: CONDITIONING_OPERATIONS, a: torch.Tensor, b: torch.Tensor | None, alpha: float):
    if b is None:
        b = torch.zeros_like(a)
    original_dtype = a.dtype
    a, b, = a.to(dtype=torch.float32), b.to(dtype=torch.float32)
    embeds: torch.Tensor = torch.zeros_like(a)

    if operation != "APPEND" and a.shape != b.shape:
        raise ValueError(f"Conditioning A: {a.shape} does not match Conditioning B: {b.shape}")

    match operation:
        case "ADD":
            torch.add(a, b, alpha=alpha, out=embeds)
        case "SUB":
            torch.sub(a, b, alpha=alpha, out=embeds)
        case "LERP":
            torch.lerp(a, b, alpha, out=embeds)
        case "PERP":
            # https://github.com/Perp-Neg/Perp-Neg-stablediffusion/blob/main/perpneg_diffusion/perpneg_stable_diffusion/pipeline_perpneg_stable_diffusion.py
            # x - ((torch.mul(x, y).sum())/(torch.norm(y)**2)) * y
            embeds = torch_funcs.perp(a, b).detach().clone()
        case "PROJ":
            embeds = torch_funcs.proj(a, b).detach().clone()
        case "APPEND":
            embeds = torch.cat((a, b), dim=1)
    return embeds.to(dtype=original_dtype)


class NamedConditioningField(Protocol):
    conditioning_name:str


def _load_conditioning(context: InvocationContext, field: NamedConditioningField) -> (
    BasicConditioningInfo
    | SDXLConditioningInfo
    | FLUXConditioningInfo
    | SD3ConditioningInfo
    | CogView4ConditioningInfo
    | None
):
    if field is None:
        return None
    else:
        return context.conditioning.load(field.conditioning_name).conditionings[0].to("cpu")


@invocation(
    "Conditioning_Math",
    title="Conditioning Math",
    tags=["math", "conditioning", "prompt", "blend", "interpolate", "append", "perpendicular", "projection"],
    category="math",
    version="1.0.0",
)
class ConditioningMathInvocation(BaseInvocation):
    """Compute between two conditioning latents"""
    
    a: ConditioningField = InputField(
        description="Conditioning A",
        input=Input.Connection, #A is required for extra information in some operations
        ui_order=0,
    )
    b: ConditioningField = InputField(
        description="Conditioning B",
        default=None,
        ui_order=1,
    )
    alpha: float = InputField(
        default=1,
        description="Alpha value for interpolation and scaling",
        title="α [optional]",
        ge=0.0,
        ui_order=3,
    )
    operation: CONDITIONING_OPERATIONS = InputField(
        default="LERP", description="The operation to perform", ui_choice_labels=CONDITIONING_OPERATIONS_LABELS,
        input=Input.Direct,
        ui_order=2,
    )


    @torch.inference_mode()
    def invoke(self, context: InvocationContext) -> ConditioningOutput:
        self.check_matching_type(context)
        conditioning_A = _load_conditioning(context, self.a)
        conditioning_B = _load_conditioning(context, self.b)

        cA: torch.Tensor = conditioning_A.embeds
        cB = conditioning_B.embeds if conditioning_B else None
        embeds = apply_operation(self.operation, cA, cB, self.alpha)

        if isinstance(conditioning_A, SDXLConditioningInfo):
            pooled_embeds = conditioning_A.pooled_embeds
            pooled_B = conditioning_B.pooled_embeds if conditioning_B else None
            pooled_embeds = apply_operation(self.operation, pooled_embeds, pooled_B, self.alpha)

            conditioning_info = SDXLConditioningInfo(
                embeds=embeds,
                pooled_embeds=pooled_embeds,
                add_time_ids=conditioning_A.add_time_ids, #always from A, just includes size information
            )
        else:
            conditioning_info = BasicConditioningInfo(embeds=embeds)

        conditioning_data = ConditioningFieldData(conditionings=[conditioning_info])
        conditioning_name = context.conditioning.save(conditioning_data)
        return ConditioningOutput(
            conditioning=ConditioningField(conditioning_name=conditioning_name)
        )

    def check_matching_type(self, context):
        if self.b is None:
            return
        conditioning_A = _load_conditioning(context, self.a)
        conditioning_B = _load_conditioning(context, self.b)
        # check that inputs are the same type
        if type(conditioning_A) != type(conditioning_B):
            raise ValueError(
                f"Conditioning A: {type(conditioning_A)} does not match Conditioning B: {type(conditioning_B)}"
            )


@invocation(
    "FLUX_Conditioning_Math",
    title="Conditioning Math - FLUX",
    tags=["math", "conditioning", "prompt", "blend", "interpolate", "append", "perpendicular", "projection"],
    category="math",
    version="1.0.1",
)
class FluxConditioningMathInvocation(ConditioningMathInvocation):
    a: FluxConditioningField = InputField(
        description="Conditioning A",
        input=Input.Connection, #A is required for extra information in some operations
        ui_order=0,
    )
    b: FluxConditioningField = InputField(
        description="Conditioning B",
        default=None,
        ui_order=1,
    )

    def invoke(self, context: InvocationContext) -> FluxConditioningOutput:
        conditioning_A = _load_conditioning(context, self.a)
        conditioning_B = _load_conditioning(context, self.b)
        clip_a: torch.Tensor = conditioning_A.clip_embeds
        clip_b: torch.Tensor = conditioning_B.clip_embeds if conditioning_B else None
        clip_embeds = apply_operation(self.operation, clip_a, clip_b, self.alpha)

        t5_a: torch.Tensor = conditioning_A.t5_embeds
        t5_b: torch.Tensor = conditioning_B.t5_embeds if conditioning_B else None
        t5_embeds = apply_operation(self.operation, t5_a, t5_b, self.alpha)
        conditioning_info = FLUXConditioningInfo(clip_embeds, t5_embeds)
        conditioning_data = ConditioningFieldData(conditionings=[conditioning_info])
        conditioning_name = context.conditioning.save(conditioning_data)
        return FluxConditioningOutput(
            conditioning=FluxConditioningField(conditioning_name=conditioning_name)
        )


@invocation(
    "FLUX_Conditioning_Freeform_Math",
    title="Conditioning Math Freeform - FLUX",
    tags=["math", "conditioning", "prompt", "blend", "interpolate", "append", "perpendicular", "projection"],
    category="math",
    version="1.0.2",
)
class FluxConditioningFreeformMathInvocation(BaseInvocation):
    c1: FluxConditioningField = InputField(
        description="Conditioning 1",
        title="c1",
        input=Input.Connection,
    )
    c2: FluxConditioningField | None = InputField(
        description="Conditioning 2", title="c2", default=None
    )
    c3: FluxConditioningField | None = InputField(
        description="Conditioning 3", title="c3", default=None,
    )
    c4: FluxConditioningField | None = InputField(
        description="Conditioning 4", title="c4", default=None,
    )
    c5: FluxConditioningField | None = InputField(
        description="Conditioning 5", title="c5", default=None,
    )
    a: float = InputField(default=1, title="a")
    b: float = InputField(default=0, title="b")

    formula: str = InputField(description="Formula to apply to conditionings c1–c5. proj, perp, and all torch functions available.",
                              default="c1")


    def invoke(self, context: InvocationContext) -> FluxConditioningOutput:
        func = self._func_from_string(self.formula)

        clip_1 = self._clip_embeds(context, self.c1)
        clip_2 = self._clip_embeds(context, self.c2, clip_1)
        clip_3 = self._clip_embeds(context, self.c3, clip_1)
        clip_4 = self._clip_embeds(context, self.c4, clip_1)
        clip_5 = self._clip_embeds(context, self.c5, clip_1)
        clip_embeds = func(clip_1, clip_2, clip_3, clip_4, clip_5, self.a, self.b)

        t5_1 = self._t5_embeds(context, self.c1)
        t5_2 = self._t5_embeds(context, self.c2, t5_1)
        t5_3 = self._t5_embeds(context, self.c3, t5_1)
        t5_4 = self._t5_embeds(context, self.c4, t5_1)
        t5_5 = self._t5_embeds(context, self.c5, t5_1)
        t5_embeds = func(t5_1, t5_2, t5_3, t5_4, t5_5, self.a, self.b)

        conditioning_info = FLUXConditioningInfo(clip_embeds, t5_embeds)
        conditioning_data = ConditioningFieldData(conditionings=[conditioning_info])
        conditioning_name = context.conditioning.save(conditioning_data)
        return FluxConditioningOutput(
            conditioning=FluxConditioningField(conditioning_name=conditioning_name)
        )

    def _clip_embeds(
        self, context: InvocationContext, field: FluxConditioningField | None, like: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        cond = _load_conditioning(context, field)
        if cond:
            return cond.clip_embeds
        if like is not None:
            return torch.zeros_like(like)
        return None

    def _t5_embeds(
        self, context: InvocationContext, field: FluxConditioningField | None, like: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        cond = _load_conditioning(context, field)
        if cond:
            return cond.t5_embeds
        if like is not None:
            return torch.zeros_like(like)
        return None

    def _func_from_string(self, formula: str):
        return sympy.lambdify(
            sympy.symbols("c1 c2 c3 c4 c5 a b"),
            sympy.sympify(formula),
            [torch_funcs.functions, torch],
        )


@invocation(
    "FLUX_Redux_Conditioning_Freeform_Math",
    title="Conditioning Math Freeform - FLUX Redux",
    tags=["math", "conditioning", "prompt", "blend", "interpolate", "append", "perpendicular", "projection"],
    category="math",
    version="1.0.0",
)
class FluxReduxConditioningFreeformMathInvocation(BaseInvocation):
    c1: FluxReduxConditioningField = InputField(
        description="Conditioning 1",
        title="c1",
        input=Input.Connection,
    )
    c2: FluxReduxConditioningField | None = InputField(
        description="Conditioning 2", title="c2", default=None
    )
    c3: FluxReduxConditioningField | None = InputField(
        description="Conditioning 3", title="c3", default=None,
    )
    c4: FluxReduxConditioningField | None = InputField(
        description="Conditioning 4", title="c4", default=None,
    )
    c5: FluxReduxConditioningField | None = InputField(
        description="Conditioning 5", title="c5", default=None,
    )
    a: float = InputField(default=1, title="a")
    b: float = InputField(default=0, title="b")

    formula: str = InputField(description="Formula to apply to conditionings c1–c5. proj, perp, and all torch functions available.",
                              default="c1")


    def invoke(self, context: InvocationContext) -> FluxReduxOutput:
        func = self._func_from_string(self.formula)

        redux_1 = self._redux_embeds(context, self.c1)
        redux_2 = self._redux_embeds(context, self.c2, redux_1)
        redux_3 = self._redux_embeds(context, self.c3, redux_1)
        redux_4 = self._redux_embeds(context, self.c4, redux_1)
        redux_5 = self._redux_embeds(context, self.c5, redux_1)
        redux_embeds = func(redux_1, redux_2, redux_3, redux_4, redux_5, self.a, self.b)

        conditioning_name = context.tensors.save(redux_embeds)
        return FluxReduxOutput(
            redux_cond=FluxReduxConditioningField(conditioning=TensorField(tensor_name=conditioning_name))
        )


    def _redux_embeds(
        self, context: InvocationContext, field: FluxReduxConditioningField | None, like: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        if field:
            return context.tensors.load(field.conditioning.tensor_name)
        if like is not None:
            return torch.zeros_like(like)
        return None


    def _func_from_string(self, formula: str):
        return sympy.lambdify(
            sympy.symbols("c1 c2 c3 c4 c5 a b"),
            sympy.sympify(formula),
            [torch_funcs.functions, torch],
        )


@invocation(
    "CogView4_Conditioning_Math",
    title="Conditioning Math - CogView4",
    tags=["math", "conditioning", "prompt", "blend", "interpolate", "append", "perpendicular", "projection"],
    category="math",
    version="1.0.1",
)
class CogView4ConditioningMathInvocation(ConditioningMathInvocation):
    a: CogView4ConditioningField = InputField(
        description="Conditioning A",
        input=Input.Connection, #A is required for extra information in some operations
        ui_order=0,
    )
    b: CogView4ConditioningField = InputField(
        description="Conditioning B",
        default=None,
        ui_order=1,
    )

    def invoke(self, context: InvocationContext) -> CogView4ConditioningOutput:
        conditioning_A = _load_conditioning(context, self.a)
        conditioning_B = _load_conditioning(context, self.b)
        glm_a: torch.Tensor = conditioning_A.glm_embeds
        glm_b: torch.Tensor = conditioning_B.glm_embeds if conditioning_B else None
        glm_embeds = apply_operation(self.operation, glm_a, glm_b, self.alpha)

        conditioning_info = CogView4ConditioningInfo(glm_embeds)
        conditioning_data = ConditioningFieldData(conditionings=[conditioning_info])
        conditioning_name = context.conditioning.save(conditioning_data)
        return CogView4ConditioningOutput(
            conditioning=CogView4ConditioningField(conditioning_name=conditioning_name)
        )


@invocation_output("extended_conditioning_output")
class ExtendedConditioningOutput(BaseInvocationOutput):
    """Base class for nodes that output a single conditioning tensor"""

    conditioning: ConditioningField = OutputField(description=FieldDescriptions.cond)
    mean: float = OutputField(description="Mean of conditioning")
    variance: float = OutputField(description="Standard deviation of conditioning")
    token_length: int = OutputField(description="Length of each token in the conditioning")
    token_space: int = OutputField(description="Number of tokens in the conditioning")



NORMALIZE_OPERATIONS = Literal[
    "INFO",
    "MEAN",
    "VAR",
    "MEAN_VAR",
]


NORMALIZE_OPERATIONS_LABELS = {
    "INFO": "Get Info (do nothing)",
    "MEAN": "Normalize Mean",
    "VAR": "Normalize Variance",
    "MEAN_VAR": "Normalize Mean and Variance",
}


@invocation(
    "normalize_conditioning",
    title="Normalize Conditioning",
    tags=["math", "conditioning", "normalize", "info", "mean", "variance"],
    category="math",
    version="2.0.0",
)
class NormalizeConditioningInvocation(BaseInvocation):
    """Normalize a conditioning (SD1.5) latent to have a mean and variance similar to another conditioning latent"""
    
    conditioning: ConditioningField = InputField(
        description="Conditioning",
        input=Input.Connection,
    )
    operation: NORMALIZE_OPERATIONS = InputField(
        default="INFO", description="The operation to perform", ui_choice_labels=NORMALIZE_OPERATIONS_LABELS,
        input=Input.Direct
    )
    mean: float = InputField(
        default=-0.1,
        description="Mean to normalize to"
    )
    var: float = InputField(
        default=1.0,
        description="Standard Deviation to normalize to",
        title="Variance"
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ExtendedConditioningOutput:
        conditioning = context.conditioning.load(self.conditioning.conditioning_name)
        c = conditioning.conditionings[0].embeds.detach().clone().to("cpu")

        mean_c, std_c, var_c = torch.mean(c), torch.std(c), torch.var(c)

        if self.operation == "INFO":
            pass
        elif self.operation == "MEAN":
            c = c * self.mean / mean_c
        elif self.operation == "VAR":
            c = ((c - mean_c) * self.var / std_c) + mean_c
        elif self.operation == "MEAN_VAR":
            c = ((c - mean_c) * np.sqrt(self.var) / std_c) + self.mean
        
        mean_out, std_out, var_out = torch.mean(c), torch.std(c), torch.var(c)

        conditioning_data = ConditioningFieldData(
            conditionings=[BasicConditioningInfo(embeds=c)]
        )

        conditioning_name = context.conditioning.save(conditioning_data)

        return ExtendedConditioningOutput(
            conditioning=ConditioningField(conditioning_name=conditioning_name),
            mean=mean_out,
            variance=var_out,
            token_length=c.shape[2],
            token_space=c.shape[1],
        )
