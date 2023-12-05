from typing import Literal

import numpy as np
import torch
from pydantic import ValidationInfo, field_validator

from invokeai.app.invocations.primitives import FloatOutput, IntegerOutput, ConditioningField, ConditioningOutput
from invokeai.app.shared.fields import FieldDescriptions
from invokeai.app.invocations.compel import ConditioningFieldData, BasicConditioningInfo

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    InputField,
    Input,
    InvocationContext,
    invocation,
    OutputField,
    invocation_output,
    BaseInvocationOutput
)



CONDITIONING_OPERATIONS = Literal[
    "LERP",
    "APPEND",
    "ADD",
    "SUB",
    #"SLERP", #NOT IMPLEMENTED in torch at this time. May be worth writing our own method
    "PERP",
    "PROJ",
    "BLEFT",
    "BRIGHT",
]


CONDITIONING_OPERATIONS_LABELS = {
    "LERP": "Linear Interpolation A->B",
    "APPEND": "Append [A, B]",
    "ADD": "Add A+αB",
    "SUB": "Subtract A-αB",
    #"SLERP": "Spherical Interpolation A~>B",
    "PERP": "Perpendicular A⊥B",
    "PROJ": "Projection A||B",
    "BLEFT": "Buffer Left [Unc, B]",
    "BRIGHT": "Buffer Right [B, Unc]"
}


CONDITIONING_FORMATS = Literal[
    "SD1",
    "SD2",
    "SDXL"
]


@invocation(
    "SD_1.X_Conditioning_Math",
    title="SD 1.X Conditioning Math",
    tags=["math", "conditioning", "prompt", "blend", "interpolate", "append", "perpendicular", "projection"],
    category="math",
    version="1.0.0",
)
class SD1XConditioningMathInvocation(BaseInvocation):
    """Compute between two conditioning latents"""
    
    operation: CONDITIONING_OPERATIONS = InputField(
        default="LERP", description="The operation to perform", ui_choice_labels=CONDITIONING_OPERATIONS_LABELS
    )
    a: ConditioningField = InputField(
        description="Conditioning A",
        default=None
    )
    b: ConditioningField = InputField(
        description="Conditioning B",
        input=Input.Connection, #B is required for scaling operations
    )
    empty_conditioning: ConditioningField = InputField(
        description="Optional: Result of an empty prompt conditioning. Used to pad the inputs if they are different sizes. Leave blank to use zeros (SDXL).",
        title="[optional] Empty tensor",
        default = None,
    )
    alpha: float = InputField(
        default=1,
        description="Alpha value for interpolation and scaling",
        ge=0.0
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ConditioningOutput:

        conditioning_B = context.services.latents.get(self.b.conditioning_name)
        cB: torch.Tensor = conditioning_B.conditionings[0].embeds.detach().clone().to("cpu")
        if self.a is None:
            cA = torch.zeros_like(cB).to(cB.device)
        else:
            conditioning_A = context.services.latents.get(self.a.conditioning_name)
            cA = conditioning_A.conditionings[0].embeds.detach().clone().to("cpu")

        if self.empty_conditioning is None:
            cUnc = torch.zeros_like(cB).to(cB.device)
        else:
            conditioning_Unc = context.services.latents.get(self.empty_conditioning.conditioning_name)
            cUnc = conditioning_Unc.conditionings[0].embeds.detach().clone().to("cpu")
        
        shape_A = cA.shape
        shape_B = cB.shape
        shape_Unc = cUnc.shape
        if not shape_A[2] == shape_B[2] == shape_Unc[2]:
            raise ValueError(f"Conditioning A: {shape_A} does not match Conditioning B: {shape_B} or Unc: {shape_Unc}")
        


        cOut = torch.zeros_like(cA).to(cB.device)

        mean_A, std_A, var_A = torch.mean(cA), torch.std(cA), torch.var(cA)
        print(f"Conditioning A: Mean: {mean_A}, Std: {std_A}, Var: {var_A}")
        mean_B, std_B, var_B = torch.mean(cB), torch.std(cB), torch.var(cB)
        print(f"Conditioning B: Mean: {mean_B}, Std: {std_B}, Var: {var_B}")

        if self.operation == "ADD":
            torch.add(cA, cB, alpha=self.alpha, out=cOut)
        elif self.operation == "SUB":
            torch.sub(cA, cB, alpha=self.alpha, out=cOut)
        elif self.operation == "LERP":
            torch.lerp(cA, cB, self.alpha, out=cOut)
        #elif self.operation == "SLERP":
            #torch.slerp(cA, cB, self.alpha, out=cOut)
        elif self.operation == "PERP":
            # https://github.com/Perp-Neg/Perp-Neg-stablediffusion/blob/main/perpneg_diffusion/perpneg_stable_diffusion/pipeline_perpneg_stable_diffusion.py
            #x - ((torch.mul(x, y).sum())/(torch.norm(y)**2)) * y
            cOut = (cA - ((torch.mul(cA, cB).sum())/(torch.norm(cB)**2)) * cB).detach().clone()
        elif self.operation == "PROJ":
            cOut = (((torch.mul(cA, cB).sum())/(torch.norm(cB)**2)) * cB).detach().clone()
        elif self.operation == "APPEND":
            cOut = torch.cat((cA, cB), dim=1)
        elif self.operation == "BLEFT":
            cOut = torch.cat((cUnc, cB), dim=1)
        elif self.operation == "BRIGHT":
            cOut = torch.cat((cB, cUnc), dim=1)

        conditioning_data = ConditioningFieldData(
            conditionings=[
                BasicConditioningInfo(
                    embeds=cOut,
                    extra_conditioning=None,
                )
            ]
        )

        conditioning_name = f"{context.graph_execution_state_id}_{self.id}_conditioning"
        context.services.latents.save(conditioning_name, conditioning_data)

        return ConditioningOutput(
            conditioning=ConditioningField(
                conditioning_name=conditioning_name,
            ),
        )


@invocation_output("extended_conditioning_output")
class ExtendedConditioningOutput(BaseInvocationOutput):
    """Base class for nodes that output a single conditioning tensor"""

    conditioning: ConditioningField = OutputField(description=FieldDescriptions.cond)
    mean: float = OutputField(description="Mean of conditioning")
    variance: float = OutputField(description="Standard deviation of conditioning")
    token_length: int = OutputField(description="Length of each token in the conditioning")
    token_space: int = OutputField(description="Number of tokens in the conditioning")
    tokens_used: int = OutputField(description="Number of tokens used in the conditioning")



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
    version="1.0.0",
)
class NormalizeConditioningInvocation(BaseInvocation):
    """Normalize a conditioning latent to have a mean and variance similar to another conditioning latent"""
    
    conditioning: ConditioningField = InputField(
        description="Conditioning"
    )
    operation: NORMALIZE_OPERATIONS = InputField(
        default="INFO", description="The operation to perform", ui_choice_labels=NORMALIZE_OPERATIONS_LABELS
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
        conditioning = context.services.latents.get(self.conditioning.conditioning_name)
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
            conditionings=[
                BasicConditioningInfo(
                    embeds=c,
                    extra_conditioning=None,
                )
            ]
        )

        conditioning_name = f"{context.graph_execution_state_id}_{self.id}_conditioning"
        context.services.latents.save(conditioning_name, conditioning_data)

        return ExtendedConditioningOutput(
            conditioning=ConditioningField(
                conditioning_name=conditioning_name,
            ),
            mean=mean_out,
            variance=var_out,
            token_length=c.shape[2],
            token_space=c.shape[1],
            tokens_used=conditioning.conditionings[0].extra_conditioning.tokens_count_including_eos_bos
        )