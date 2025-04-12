from typing import Literal

import numpy as np
import torch

from invokeai.app.invocations.fields import FluxConditioningField
from invokeai.app.invocations.primitives import FluxConditioningOutput
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
            embeds = (a - (
                    (torch.mul(a, b).sum()) / (torch.norm(b) ** 2)) * b).detach().clone()
        case "PROJ":
            embeds = (((torch.mul(a, b).sum()) / (torch.norm(b) ** 2)) * b).detach().clone()
        case "APPEND":
            embeds = torch.cat((a, b), dim=1)
    return embeds.to(dtype=original_dtype)


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
        conditioning_A = self._load_conditioning(context, self.a)
        conditioning_B = self._load_conditioning(context, self.b)

        match conditioning_A:
            case SDXLConditioningInfo():
                cA: torch.Tensor = conditioning_A.embeds
                cB = conditioning_B.embeds if conditioning_B else None
                embeds = apply_operation(self.operation, cA, cB, self.alpha)

                pooled_embeds = conditioning_A.pooled_embeds
                pooled_B = conditioning_B.pooled_embeds if conditioning_B else None
                pooled_embeds = apply_operation(self.operation, pooled_embeds, pooled_B, self.alpha)

                conditioning_info = SDXLConditioningInfo(
                    embeds=embeds,
                    pooled_embeds=pooled_embeds,
                    add_time_ids=conditioning_A.add_time_ids, #always from A, just includes size information
                )
            case BasicConditioningInfo():
                cA: torch.Tensor = conditioning_A.embeds
                cB = conditioning_B.embeds if conditioning_B else None
                embeds = apply_operation(self.operation, cA, cB, self.alpha)
                conditioning_info = BasicConditioningInfo(embeds=embeds)
            case FLUXConditioningInfo():
                clip_a: torch.Tensor = conditioning_A.clip_embeds
                clip_b: torch.Tensor = conditioning_B.clip_embeds if conditioning_B else None
                clip_embeds = apply_operation(self.operation, clip_a, clip_b, self.alpha)

                t5_a: torch.Tensor = conditioning_A.t5_embeds
                t5_b: torch.Tensor = conditioning_B.t5_embeds if conditioning_B else None
                t5_embeds = apply_operation(self.operation, t5_a, t5_b, self.alpha)
                conditioning_info = FLUXConditioningInfo(clip_embeds, t5_embeds)
            case CogView4ConditioningInfo():
                glm_a: torch.Tensor = conditioning_A.glm_embeds
                glm_b: torch.Tensor = conditioning_B.glm_embeds if conditioning_B else None
                glm_embeds = apply_operation(self.operation, glm_a, glm_b, self.alpha)
                conditioning_info = CogView4ConditioningInfo(glm_embeds)
            case SD3ConditioningInfo():
                raise NotImplementedError("TODO: SD3")
            case _:
                raise NotImplementedError(f"Unknown conditioning info {conditioning_A}")

        conditioning_data = ConditioningFieldData(conditionings=[conditioning_info])
        conditioning_name = context.conditioning.save(conditioning_data)

        return ConditioningOutput(
            conditioning=ConditioningField(
                conditioning_name=conditioning_name,
            )
        )

    def _load_conditioning(
        self, context: InvocationContext, field: ConditioningField
    ) -> (
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


    def check_matching_type(self, context):
        if self.b is None:
            return
        conditioning_A = self._load_conditioning(context, self.a)
        conditioning_B = self._load_conditioning(context, self.b)
        # check that inputs are the same type
        if type(conditioning_A) != type(conditioning_B):
            raise ValueError(
                f"Conditioning A: {type(conditioning_A)} does not match Conditioning B: {type(conditioning_B)}"
            )


@invocation(
    "Conditioning_Math_FLUX",
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
        conditioning_A = self._load_conditioning(context, self.a)
        conditioning_B = self._load_conditioning(context, self.b)
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
            conditioning=FluxConditioningField(
                conditioning_name=conditioning_name
            )
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
