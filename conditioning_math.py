from typing import Literal

import numpy as np
import torch

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


    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ConditioningOutput:
        conditioning_A: BasicConditioningInfo = context.conditioning.load(self.a.conditioning_name).conditionings[0]
        cA: torch.Tensor = conditioning_A.embeds.detach().clone().to("cpu")
        dt = cA.dtype
        cA = cA.to(torch.float32)
        if self.b is None:
            cB = torch.zeros_like(cA)
        else:
            conditioning_B: BasicConditioningInfo = context.conditioning.load(self.b.conditioning_name).conditionings[0]
            cB = conditioning_B.embeds.detach().clone().to("cpu", dtype=torch.float32)
        
            #check that inputs are the same type
            if type(conditioning_A) != type(conditioning_B):
                raise ValueError(f"Conditioning A: {type(conditioning_A)} does not match Conditioning B: {type(conditioning_B)}")
        
        shape_A = cA.shape
        shape_B = cB.shape

        if (shape_A != shape_B) and (self.operation != "APPEND"):
            raise ValueError(f"Conditioning A: {shape_A} does not match Conditioning B: {shape_B}")
        
        if type(conditioning_A) == BasicConditioningInfo: #NOT SDXL
            embeds: torch.Tensor = torch.zeros_like(cA)
            # extra_conditioning = conditioning_A.extra_conditioning
            # ec_A_tokens = extra_conditioning.tokens_count_including_eos_bos
            # ec_A_cross_attention = extra_conditioning.cross_attention_control_args

            # if self.b is None:
            #     ec_B_tokens = 0
            # else:
            #     ec_B_tokens = conditioning_B.extra_conditioning.tokens_count_including_eos_bos

            # ec_tokens = max(ec_A_tokens, ec_B_tokens) #not sure if this is ever used, but this should be a safe assumption

            if self.operation == "ADD":
                torch.add(cA, cB, alpha=self.alpha, out=embeds)
            elif self.operation == "SUB":
                torch.sub(cA, cB, alpha=self.alpha, out=embeds)
            elif self.operation == "LERP":
                torch.lerp(cA, cB, self.alpha, out=embeds)
            elif self.operation == "PERP":
                # https://github.com/Perp-Neg/Perp-Neg-stablediffusion/blob/main/perpneg_diffusion/perpneg_stable_diffusion/pipeline_perpneg_stable_diffusion.py
                #x - ((torch.mul(x, y).sum())/(torch.norm(y)**2)) * y
                embeds = (cA - ((torch.mul(cA, cB).sum())/(torch.norm(cB)**2)) * cB).detach().clone()
            elif self.operation == "PROJ":
                embeds = (((torch.mul(cA, cB).sum())/(torch.norm(cB)**2)) * cB).detach().clone()
            elif self.operation == "APPEND":
                embeds = torch.cat((cA, cB), dim=1)
                # ec_tokens = cA.shape[1] + ec_B_tokens #append is the only time this changes

            conditioning_data = ConditioningFieldData(
                conditionings=[
                    BasicConditioningInfo(
                        embeds=embeds.to(dtype=dt),
                        # extra_conditioning=ExtraConditioningInfo(
                        #     tokens_count_including_eos_bos=ec_tokens,
                        #     cross_attention_control_args=ec_A_cross_attention, #not going to bother with cross attention control for now
                        # ),
                    )
                ]
            )
        else: #SDXL
            embeds = torch.zeros_like(cA).to(cA.device)
            # ec_A_tokens = conditioning_A.extra_conditioning.tokens_count_including_eos_bos
            # ec_A_cross_attention = conditioning_A.extra_conditioning.cross_attention_control_args
            pooled_embeds = conditioning_A.pooled_embeds.detach().clone().to("cpu", dtype=torch.float32) #default for operations that don't affect it
            add_time_ids = conditioning_A.add_time_ids.detach().clone().to("cpu") #default for operations that don't affect it

            if self.b is None:
                pooled_B = torch.zeros_like(pooled_embeds).to("cpu", dtype=torch.float32)
                add_time_ids_B = torch.zeros_like(add_time_ids).to("cpu")
                # ec_B_tokens = 0
            else:
                pooled_B = conditioning_B.pooled_embeds.detach().clone().to("cpu", dtype=torch.float32)
                add_time_ids_B = conditioning_B.add_time_ids.detach().clone().to("cpu")
                # ec_B_tokens = conditioning_B.extra_conditioning.tokens_count_including_eos_bos
            
            # ec_tokens = max(ec_A_tokens, ec_B_tokens) #not sure if this is ever used, but this should be a safe assumption

            if self.operation == "ADD":
                torch.add(cA, cB, alpha=self.alpha, out=embeds)
                torch.add(pooled_embeds, pooled_B, alpha=self.alpha, out=pooled_embeds)
            elif self.operation == "SUB":
                torch.sub(cA, cB, alpha=self.alpha, out=embeds)
                torch.sub(pooled_embeds, pooled_B, alpha=self.alpha, out=pooled_embeds)
            elif self.operation == "LERP":
                torch.lerp(cA, cB, self.alpha, out=embeds)
                torch.lerp(pooled_embeds, pooled_B, self.alpha, out=pooled_embeds)
            elif self.operation == "PERP":
                # https://github.com/Perp-Neg/Perp-Neg-stablediffusion/blob/main/perpneg_diffusion/perpneg_stable_diffusion/pipeline_perpneg_stable_diffusion.py
                #x - ((torch.mul(x, y).sum())/(torch.norm(y)**2)) * y
                embeds = (cA - ((torch.mul(cA, cB).sum())/(torch.norm(cB)**2)) * cB).detach().clone()
                pooled_embeds = (pooled_embeds - ((torch.mul(pooled_embeds, pooled_B).sum())/(torch.norm(pooled_B)**2)) * pooled_B).detach().clone()
            elif self.operation == "PROJ":
                embeds = (((torch.mul(cA, cB).sum())/(torch.norm(cB)**2)) * cB).detach().clone()
                pooled_embeds = (((torch.mul(pooled_embeds, pooled_B).sum())/(torch.norm(pooled_B)**2)) * pooled_B).detach().clone()
            elif self.operation == "APPEND":
                embeds = torch.cat((cA, cB), dim=1)
                # ec_tokens = cA.shape[1] + ec_B_tokens #append is the only time this changes

            conditioning_data = ConditioningFieldData(
                conditionings=[
                    SDXLConditioningInfo(
                        embeds=embeds.to(dtype=dt),
                        # extra_conditioning=ExtraConditioningInfo(
                        #     tokens_count_including_eos_bos=ec_tokens,
                        #     cross_attention_control_args=ec_A_cross_attention, #not going to bother with cross attention control for now
                        # ),
                        pooled_embeds=pooled_embeds.to(dtype=dt),
                        add_time_ids=add_time_ids, #always from A, just includes size information
                    )
                ]
            )
        
        conditioning_name = context.conditioning.save(conditioning_data)

        return ConditioningOutput(
            conditioning=ConditioningField(
                conditioning_name=conditioning_name,
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
            conditionings=[
                BasicConditioningInfo(
                    embeds=c,
                    # extra_conditioning=None,
                )
            ]
        )

        conditioning_name = context.conditioning.save(conditioning_data)

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
