"""Export SAM 3.1 multiplex checkpoint into ONNX subgraphs for browser inference.

Outputs (default: <checkpoint_dir>/onnx):
  vision-encoder.onnx      image -> detector FPN + PVS embeddings
  text-encoder.onnx        CLIP tokens -> language features
  grounding-decoder.onnx   FPN + text + box/point exemplars -> PCS masks
  prompt-decoder.onnx      PVS embeddings + points/boxes/mask -> instance masks
  clip_bpe.json            tokenizer tables matching official SimpleTokenizer
  io_spec.json             tensor names / shapes used by yolo-onnx-web
"""

from __future__ import annotations

import argparse
import importlib.abc
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

os.environ["USE_PERFLIB"] = "0"
os.environ["DNNL_DEFAULT_FPMATH_MODE"] = "STRICT"
os.environ["ONEDNN_DEFAULT_FPMATH_MODE"] = "STRICT"

import torch
import torch.nn as nn
import torch.nn.functional as F

# Tracing SAM3 hits a few Python-only asserts / profilers.
if hasattr(torch, "_assert_async"):
    torch._assert_async = lambda *args, **kwargs: None  # type: ignore[assignment]


_orig_sdpa = torch.nn.functional.scaled_dot_product_attention


def _sdpa_keep_query_dtype(q, k, v, *args, **kwargs):
    output = _orig_sdpa(q, k, v, *args, **kwargs)
    if output.dtype != q.dtype:
        output = output.to(dtype=q.dtype)
    return output


torch.nn.functional.scaled_dot_product_attention = _sdpa_keep_query_dtype  # type: ignore[assignment]


def _patch_fused_mlp_for_onnx() -> None:
    """ViT MLP uses perflib.addmm_act which always casts to bfloat16.

    That breaks ONNX export (bf16 activations vs fp32 Linear). Replace it with
    a plain Linear + GELU/ReLU path before sam3.model.vitdet is imported.
    """
    import sam3.perflib.fused as fused

    def addmm_act(activation, linear, mat1):
        y = linear(mat1)
        if activation in (F.relu, torch.nn.ReLU):
            return F.relu(y)
        if activation in (F.gelu, torch.nn.GELU):
            return F.gelu(y)
        raise ValueError(f"Unexpected activation {activation}")

    fused.addmm_act = addmm_act
    vitdet = sys.modules.get("sam3.model.vitdet")
    if vitdet is not None:
        vitdet.addmm_act = addmm_act


class _StubEdtLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        def edt_triton(masks, *args, **kwargs):
            return masks

        module.edt_triton = edt_triton


class _StubEdtFinder(importlib.abc.MetaPathFinder):
    """Avoid importing Triton via sam3.model.edt on Windows."""

    def find_spec(self, fullname, path, target=None):
        if fullname == "sam3.model.edt":
            return importlib.util.spec_from_loader(fullname, _StubEdtLoader())
        return None


sys.meta_path.insert(0, _StubEdtFinder())


def _as_tensor(value):
    if value is None:
        return None
    if hasattr(value, "tensors"):
        return value.tensors
    return value


def _nested(tensor, mask=None):
    from sam3.model.data_misc import NestedTensor

    return NestedTensor(tensor, mask)


class VisionEncoderWrapper(nn.Module):
    """Detector vision backbone + PVS high-res projections."""

    def __init__(self, detector, tracker):
        super().__init__()
        self.backbone = detector.backbone
        self.interactive_mask_decoder = tracker.interactive_sam_mask_decoder
        self.no_mem_embed = getattr(
            tracker, "interactivity_no_mem_embed", getattr(tracker, "no_mem_embed", None)
        )
        if self.no_mem_embed is None:
            raise AttributeError("Tracker is missing no_mem_embed / interactivity_no_mem_embed")
        self.bb_feat_sizes = [(288, 288), (144, 144), (72, 72)]

    def forward(self, images: torch.Tensor):
        images = images.float()
        output = self.backbone.forward_image(
            images,
            need_sam3_out=True,
            need_interactive_out=True,
            need_propagation_out=False,
        )
        det_fpn = [_as_tensor(item) for item in output["backbone_fpn"]]
        det_pos = [_as_tensor(item) for item in output["vision_pos_enc"]]

        interactive = output["interactive"]
        pvs_fpn = [_as_tensor(item) for item in interactive["backbone_fpn"]]
        pvs_fpn[0] = self.interactive_mask_decoder.conv_s0(pvs_fpn[0])
        pvs_fpn[1] = self.interactive_mask_decoder.conv_s1(pvs_fpn[1])

        vision_feats = [feat.flatten(2).permute(2, 0, 1) for feat in pvs_fpn]
        vision_feats[-1] = vision_feats[-1] + self.no_mem_embed
        pvs_maps = [
            feat.permute(1, 2, 0).view(images.shape[0], -1, height, width)
            for feat, (height, width) in zip(vision_feats[::-1], self.bb_feat_sizes[::-1])
        ][::-1]

        return (
            det_fpn[0],
            det_fpn[1],
            det_fpn[2],
            det_pos[0],
            det_pos[1],
            det_pos[2],
            pvs_maps[0],
            pvs_maps[1],
            pvs_maps[2],
        )


class TextEncoderWrapper(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.encoder = text_encoder.encoder
        self.resizer = text_encoder.resizer

    def forward(self, input_ids: torch.Tensor):
        pad_mask = input_ids == 0
        pooled_and_tokens = self.encoder(input_ids)
        if isinstance(pooled_and_tokens, tuple):
            _, tokens = pooled_and_tokens
        else:
            tokens = pooled_and_tokens
        language_features = self.resizer(tokens.transpose(0, 1))
        language_embeds = self.encoder.token_embedding(input_ids).transpose(0, 1)
        return pad_mask, language_features, language_embeds


class GroundingDecoderWrapper(nn.Module):
    """PCS decoder: text + geometric exemplars (boxes / points)."""

    def __init__(self, detector):
        super().__init__()
        self.detector = detector

    def forward(
        self,
        det_fpn_0,
        det_fpn_1,
        det_fpn_2,
        det_pos_0,
        det_pos_1,
        det_pos_2,
        language_features,
        language_mask,
        box_coords,
        box_labels,
        box_pad_mask,
        point_coords,
        point_labels,
        point_pad_mask,
    ):
        from sam3.model.data_misc import FindStage
        from sam3.model.geometry_encoders import Prompt

        backbone_out = {
            "backbone_fpn": [
                _nested(det_fpn_0),
                _nested(det_fpn_1),
                _nested(det_fpn_2),
            ],
            "vision_pos_enc": [det_pos_0, det_pos_1, det_pos_2],
            "language_features": language_features,
            "language_mask": language_mask,
        }
        find_input = FindStage(
            img_ids=torch.zeros(1, dtype=torch.long, device=det_fpn_0.device),
            text_ids=torch.zeros(1, dtype=torch.long, device=det_fpn_0.device),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        geometric_prompt = Prompt(
            box_embeddings=box_coords,
            box_mask=box_pad_mask,
            box_labels=box_labels,
            point_embeddings=point_coords,
            point_mask=point_pad_mask,
            point_labels=point_labels,
        )
        outputs = self.detector.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            find_target=None,
            geometric_prompt=geometric_prompt,
        )
        pred_masks = outputs["pred_masks"]
        pred_boxes = outputs["pred_boxes"]
        pred_logits = outputs["pred_logits"]
        presence = outputs["presence_logit_dec"]
        if pred_logits.dim() == 3:
            pred_logits = pred_logits.squeeze(-1)
        if presence.dim() > 2:
            presence = presence.reshape(presence.shape[0], -1)[:, -1]
        elif presence.dim() == 2:
            presence = presence[:, -1]
        return pred_masks, pred_boxes, pred_logits, presence


class PromptDecoderWrapper(nn.Module):
    """PVS decoder: SAM-style points / boxes / mask prompts."""

    def __init__(self, tracker):
        super().__init__()
        self.prompt_encoder = tracker.interactive_sam_prompt_encoder
        self.mask_decoder = tracker.interactive_sam_mask_decoder
        self.register_buffer("dense_pe", self.prompt_encoder.get_dense_pe())
        self.embed_dim = self.prompt_encoder.embed_dim
        self.image_embedding_size = self.prompt_encoder.image_embedding_size

    def forward(
        self,
        image_embed,
        high_res_0,
        high_res_1,
        point_coords,
        point_labels,
        mask_input,
        has_mask_input,
    ):
        sparse_embeddings, dense_from_mask = self.prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=mask_input,
        )
        no_mask = self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            image_embed.shape[0],
            -1,
            self.image_embedding_size[0],
            self.image_embedding_size[1],
        )
        has_mask = has_mask_input.view(-1, 1, 1, 1)
        dense_embeddings = dense_from_mask * has_mask + no_mask * (1.0 - has_mask)

        # Return all mask tokens: [0]=single, [1:]=multimask. JS selects.
        masks, iou_predictions, _, object_score_logits = self.mask_decoder.predict_masks(
            image_embeddings=image_embed,
            image_pe=self.dense_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            repeat_image=False,
            high_res_features=[high_res_0, high_res_1],
        )
        low_res_masks = torch.clamp(masks, -32.0, 32.0)
        return low_res_masks, iou_predictions, object_score_logits


def _move_python_tensor_caches(module: nn.Module, device: str | torch.device) -> None:
    device = torch.device(device)
    for child in module.modules():
        cache = getattr(child, "compilable_cord_cache", None)
        if isinstance(cache, tuple) and len(cache) == 2:
            child.compilable_cord_cache = tuple(item.to(device) for item in cache)
        coord_cache = getattr(child, "coord_cache", None)
        if isinstance(coord_cache, dict):
            child.coord_cache = {
                key: tuple(item.to(device) for item in value)
                if isinstance(value, tuple)
                else value
                for key, value in coord_cache.items()
            }


def build_sam3_multiplex_for_export(checkpoint_path: str, device: str):
    import pkg_resources

    _patch_fused_mlp_for_onnx()
    from sam3.model.geometry_encoders import SequenceGeometryEncoder  # noqa: F401
    from sam3.model.model_misc import DotProductScoring, MLP
    from sam3.model.sam3_multiplex_base import Sam3MultiplexPredictorWrapper
    from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector
    from sam3.model.sam3_multiplex_tracking import Sam3MultiplexTrackingWithInteractivity
    from sam3.model.vl_combiner import SAM3VLBackboneTri
    from sam3.model_builder import (
        _create_dot_product_scoring,
        _create_geometry_encoder,
        _create_multiplex_tri_backbone,
        _create_sam3_transformer,
        _create_segmentation_head,
        _create_text_encoder,
        build_sam3_multiplex_video_model,
    )

    bpe_path = pkg_resources.resource_filename("sam3", "assets/bpe_simple_vocab_16e6.txt.gz")
    tracker_model = build_sam3_multiplex_video_model(
        checkpoint_path=None,
        load_from_HF=False,
        multiplex_count=16,
        use_fa3=False,
        use_rope_real=True,
        compile=False,
        strict_state_dict_loading=False,
        device="cpu",
    )
    del tracker_model.backbone
    tracker_model.backbone = None

    sam2_predictor = Sam3MultiplexPredictorWrapper(
        model=tracker_model,
        per_obj_inference=False,
        fill_hole_area=0,
        is_multiplex=True,
        is_multiplex_dynamic=True,
    )

    tri_neck = _create_multiplex_tri_backbone(compile_mode=None, use_fa3=False, use_rope_real=True)
    text_encoder = _create_text_encoder(bpe_path)
    backbone = SAM3VLBackboneTri(scalp=0, visual=tri_neck, text=text_encoder)
    transformer = _create_sam3_transformer(use_fa3=False)
    segmentation_head = _create_segmentation_head(use_fa3=False)
    geometry_encoder = _create_geometry_encoder()
    dot_prod_scoring = _create_dot_product_scoring()

    detector = Sam3MultiplexDetector(
        num_feature_levels=1,
        backbone=backbone,
        transformer=transformer,
        segmentation_head=segmentation_head,
        semantic_segmentation_head=None,
        input_geometry_encoder=geometry_encoder,
        use_early_fusion=True,
        use_dot_prod_scoring=True,
        dot_prod_scoring=dot_prod_scoring,
        supervise_joint_box_scores=True,
        is_multiplex=True,
    )

    demo_model = Sam3MultiplexTrackingWithInteractivity(
        tracker=sam2_predictor,
        detector=detector,
        score_threshold_detection=0.4,
        det_nms_thresh=0.1,
        det_nms_use_iom=True,
        assoc_iou_thresh=0.1,
        new_det_thresh=0.65,
        hotstart_delay=15,
        hotstart_unmatch_thresh=8,
        hotstart_dup_thresh=8,
        suppress_unmatched_only_within_hotstart=False,
        suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
        suppress_det_close_to_boundary=True,
        fill_hole_area=0,
        recondition_every_nth_frame=16,
        use_iom_recondition=True,
        iom_thresh_recondition=0.5,
        masklet_confirmation_enable=True,
        reconstruction_bbox_iou_thresh=-1,
        reconstruction_bbox_det_score=0.8,
        max_num_objects=16,
        postprocess_batch_size=16,
        use_batched_grounding=True,
        batched_grounding_batch_size=16,
        max_num_kboxes=0,
        sprinkle_removal_area=0,
        is_multiplex=True,
        image_size=1008,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        compile_model=False,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    needs_remap = any(k.startswith("sam3_model.") or k.startswith("sam2_predictor.") for k in ckpt)
    if needs_remap:
        remapped = {}
        for key, value in ckpt.items():
            new_key = key
            if key.startswith("sam3_model."):
                new_key = "detector." + key[len("sam3_model.") :]
            elif key.startswith("sam2_predictor."):
                new_key = "tracker." + key[len("sam2_predictor.") :]
            remapped[new_key] = value
        ckpt = remapped

    missing, unexpected = demo_model.load_state_dict(ckpt, strict=False)
    from collections import Counter
    missing_prefixes = Counter(".".join(key.split(".")[:2]) if "." in key else key for key in missing)
    print(f"Loaded checkpoint. missing={len(missing)} unexpected={len(unexpected)}")
    print(f"Missing prefixes: {dict(missing_prefixes)}")
    if missing:
        notable = [key for key in missing if "freqs_cis" not in key]
        print(f"Missing non-RoPE keys ({len(notable)}):")
        for key in notable[:32]:
            print(f"  {key}")
    demo_model.eval()
    demo_model.to(device)
    _move_python_tensor_caches(demo_model, device)
    return demo_model, bpe_path


def dump_clip_bpe(bpe_path: str, output_path: Path):
    from sam3.model.tokenizer_ve import SimpleTokenizer

    tokenizer = SimpleTokenizer(bpe_path=bpe_path, context_length=32)
    payload = {
        "context_length": 32,
        "sot_token_id": int(tokenizer.sot_token_id),
        "eot_token_id": int(tokenizer.eot_token_id),
        "encoder": tokenizer.encoder,
        "bpe_ranks": {" ".join(pair): rank for pair, rank in tokenizer.bpe_ranks.items()},
        "byte_encoder": {str(key): value for key, value in tokenizer.byte_encoder.items()},
    }
    output_path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"Wrote tokenizer tables: {output_path}")


def _constant_bool(name: str, init_map: dict, out_map: dict, seen: set[str] | None = None) -> bool | None:
    """Resolve a tensor name to a constant boolean, following Identity/Constant."""
    from onnx import numpy_helper

    seen = seen or set()
    if not name or name in seen:
        return None
    seen.add(name)

    initializer = init_map.get(name)
    if initializer is not None:
        array = numpy_helper.to_array(initializer)
        if array.size == 1:
            return bool(array.reshape(-1)[0])
        return None

    producer = out_map.get(name)
    if producer is None:
        return None
    if producer.op_type == "Identity" and producer.input:
        return _constant_bool(producer.input[0], init_map, out_map, seen)
    if producer.op_type == "Constant":
        for attr in producer.attribute:
            if attr.name == "value":
                array = numpy_helper.to_array(attr.t)
                if array.size == 1:
                    return bool(array.reshape(-1)[0])
        return None
    return None


def fold_constant_if_nodes(model) -> int:
    """Inline ONNX If nodes whose condition is a constant.

    Torch exports ViT pad as If. The unused branch concatenates a pad spec of
    length 5 while the taken branch is length 4. Native ORT leniently merges
    that; onnxruntime-web fails session creation with ShapeInferenceError.
    """
    from copy import deepcopy

    graph = model.graph
    folded = 0
    changed = True

    while changed:
        changed = False
        init_map = {item.name: item for item in graph.initializer}
        out_map = {output: node for node in graph.node for output in node.output}
        new_nodes = []
        replacements: dict[str, str] = {}

        for node in graph.node:
            if replacements:
                for index, input_name in enumerate(list(node.input)):
                    if input_name in replacements:
                        node.input[index] = replacements[input_name]

            if node.op_type != "If":
                new_nodes.append(node)
                continue

            condition = _constant_bool(node.input[0], init_map, out_map)
            if condition is None:
                new_nodes.append(node)
                continue

            taken = None
            for attr in node.attribute:
                if attr.name == ("then_branch" if condition else "else_branch"):
                    taken = attr.g
            if taken is None:
                new_nodes.append(node)
                continue

            if len(taken.node) == 1 and taken.node[0].op_type == "Identity" and len(node.output) == 1:
                replacements[node.output[0]] = taken.node[0].input[0]
                folded += 1
                changed = True
                continue

            rename = {
                subgraph_out.name: if_out
                for subgraph_out, if_out in zip(taken.output, node.output)
            }
            existing_inits = {item.name for item in graph.initializer}
            for init in taken.initializer:
                if init.name not in existing_inits:
                    graph.initializer.append(init)
                    existing_inits.add(init.name)
            for subgraph_node in taken.node:
                copied = deepcopy(subgraph_node)
                for index, output_name in enumerate(list(copied.output)):
                    if output_name in rename:
                        copied.output[index] = rename[output_name]
                new_nodes.append(copied)
            folded += 1
            changed = True

        if replacements:
            for output in graph.output:
                if output.name in replacements:
                    output.name = replacements[output.name]

        del graph.node[:]
        graph.node.extend(new_nodes)

    return folded


def fold_onnx_file(path: Path) -> int:
    import onnx

    print(f"Folding constant If nodes in {path.name} ...")
    model = onnx.load(str(path), load_external_data=True)
    folded = fold_constant_if_nodes(model)
    if folded == 0:
        print("  no constant If nodes")
        return 0

    tmp_path = path.with_suffix(".onnx.tmp")
    onnx.save(model, str(tmp_path))
    tmp_path.replace(path)
    print(f"  inlined {folded} If node(s), saved {path}")
    return folded


def convert_initializers_to_fp16(model):
    """Store FLOAT weights as FLOAT16 and Cast back to FLOAT32 at use.

    Browser WASM is limited to ~2–4GB. Keeping compute in fp32 avoids LayerNorm
    overflow, while cutting initializer size (and session RAM) roughly in half.
    Disable ORT graph optimization when loading, or Casts may be folded back to fp32.
    """
    from onnx import TensorProto, helper
    from onnxruntime.transformers.float16 import convert_tensor_float_to_float16

    graph = model.graph
    cast_nodes = []
    converted = 0
    for initializer in graph.initializer:
        if initializer.data_type != TensorProto.FLOAT:
            continue
        original_name = initializer.name
        fp16_name = f"{original_name}__fp16"
        convert_tensor_float_to_float16(initializer)
        initializer.name = fp16_name
        cast_nodes.append(
            helper.make_node(
                "Cast",
                [fp16_name],
                [original_name],
                name=f"/fp16_cast/{original_name}",
                to=TensorProto.FLOAT,
            )
        )
        converted += 1

    if cast_nodes:
        nodes = list(graph.node)
        del graph.node[:]
        graph.node.extend(cast_nodes)
        graph.node.extend(nodes)
    return converted


def convert_onnx_file_to_fp16(source: Path, destination: Path | None = None) -> Path:
    import gc

    import onnx

    destination = destination or source.with_name(source.stem + ".fp16.onnx")
    print(f"Converting {source.name} weights to fp16 -> {destination.name} ...")
    model = onnx.load(str(source), load_external_data=True)
    converted = convert_initializers_to_fp16(model)
    tmp_path = destination.with_suffix(".onnx.tmp")
    onnx.save(model, str(tmp_path))
    tmp_path.replace(destination)
    size_mb = destination.stat().st_size / (1024 * 1024)
    print(f"  converted {converted} initializers, saved {destination} ({size_mb:.1f} MB)")
    del model
    gc.collect()
    return destination


_FP16_FILES = {
    "vision": "vision-encoder.onnx",
    "text": "text-encoder.onnx",
    "grounding": "grounding-decoder.onnx",
    "prompt": "prompt-decoder.onnx",
}


def _record_onnx_io(path: Path) -> dict:
    try:
        import onnx

        model = onnx.load(str(path), load_external_data=False)
        graph = model.graph

        def _dims(value_info):
            return [
                dim.dim_param or dim.dim_value
                for dim in value_info.type.tensor_type.shape.dim
            ]

        return {
            "file": path.name,
            "inputs": {item.name: _dims(item) for item in graph.input},
            "outputs": {item.name: _dims(item) for item in graph.output},
        }
    except Exception as exc:
        print(f"  warning: could not inspect {path.name}: {exc}")
        return {"file": path.name}


def _export(model, dummy, path: Path, input_names, output_names, dynamic_axes, opset: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {path.name} ...")
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cpu", enabled=False), torch.autocast(
        device_type="cuda", enabled=False
    ):
        print("  dry-run forward ...")
        outputs = model(*dummy) if isinstance(dummy, tuple) else model(dummy)
        if isinstance(outputs, (tuple, list)):
            for index, tensor in enumerate(outputs):
                if torch.is_tensor(tensor):
                    print(f"    out[{index}] shape={tuple(tensor.shape)} dtype={tensor.dtype}")
        print("  tracing onnx ...")
        export_kwargs = {
            "input_names": input_names,
            "output_names": output_names,
            "opset_version": opset,
            "do_constant_folding": True,
            "dynamo": False,
        }
        if dynamic_axes:
            export_kwargs["dynamic_axes"] = dynamic_axes
        torch.onnx.export(
            model,
            dummy,
            str(path),
            **export_kwargs,
        )
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  saved {path} ({size_mb:.1f} MB)")
    fold_onnx_file(path)
    return _record_onnx_io(path)


def _concat_padded_sequences_onnx(seq1, mask1, seq2, mask2, return_index: bool = False):
    """ONNX-friendly concat. Official scatter packing bakes dummy seq lengths into Reshape."""
    concatenated_sequence = torch.cat((seq1, seq2), dim=0)
    concatenated_mask = torch.cat((mask1, mask2), dim=1)
    if return_index:
        index = torch.arange(seq2.shape[0], device=seq2.device)[:, None].expand(-1, seq2.shape[1])
        index = index + seq1.shape[0]
        return concatenated_sequence, concatenated_mask, index
    return concatenated_sequence, concatenated_mask


def _encode_prompt_skip_empty_visual(
    self,
    backbone_out,
    find_input,
    geometric_prompt,
    visual_prompt_embed=None,
    visual_prompt_mask=None,
    encode_text=True,
    prev_mask_pred=None,
):
    """Same as Sam3Image._encode_prompt but do not torch.cat empty visual tensors."""
    txt_ids = find_input.text_ids
    txt_feats = backbone_out["language_features"][:, txt_ids]
    txt_masks = backbone_out["language_mask"][txt_ids]

    feat_tuple = self._get_img_feats(backbone_out, find_input.img_ids)
    backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = feat_tuple

    if prev_mask_pred is not None:
        img_feats = [img_feats[-1] + prev_mask_pred]
    geo_feats, geo_masks = self.geometry_encoder(
        geo_prompt=geometric_prompt,
        img_feats=img_feats,
        img_sizes=vis_feat_sizes,
        img_pos_embeds=img_pos_embeds,
    )
    parts = []
    masks = []
    if encode_text:
        parts.append(txt_feats)
        masks.append(txt_masks)
    parts.append(geo_feats)
    masks.append(geo_masks)
    if visual_prompt_embed is not None and visual_prompt_embed.numel() > 0:
        parts.append(visual_prompt_embed)
        masks.append(visual_prompt_mask)
    prompt = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
    prompt_mask = masks[0] if len(masks) == 1 else torch.cat(masks, dim=1)
    return prompt, prompt_mask, backbone_out


def _patch_sam3_for_onnx_export() -> None:
    """Keep prompt sequence length dynamic so N boxes / points do not freeze Reshape to 35."""
    import sam3.model.geometry_encoders as geo
    from sam3.model.sam3_image import Sam3Image

    geo.concat_padded_sequences = _concat_padded_sequences_onnx
    Sam3Image._encode_prompt = _encode_prompt_skip_empty_visual


@torch.no_grad()
def export_all(args):
    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    opset = args.opset

    print(f"Building SAM 3.1 multiplex from {checkpoint}")
    demo_model, bpe_path = build_sam3_multiplex_for_export(str(checkpoint), device="cpu")
    detector = demo_model.detector
    tracker = demo_model.tracker.model
    dump_clip_bpe(bpe_path, output_dir / "clip_bpe.json")

    modules = set(args.modules) if args.modules else {"vision", "text", "grounding", "prompt"}
    spec_path = output_dir / "io_spec.json"
    spec = {
        "image_size": 1008,
        "mask_size": 288,
        "text_length": 32,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "files": {},
    }
    if spec_path.exists():
        try:
            existing = json.loads(spec_path.read_text(encoding="utf-8"))
            spec["files"].update(existing.get("files", {}))
        except json.JSONDecodeError:
            pass

    if "text" in modules:
        wrapper = TextEncoderWrapper(detector.backbone.language_backbone).to(device).eval()
        dummy_ids = torch.randint(0, 49408, (1, 32), device=device)
        with torch.no_grad():
            pad_mask, feats, embeds = wrapper(dummy_ids.cpu() if device == "cpu" else dummy_ids)
        spec["files"]["text"] = {
            "file": "text-encoder.onnx",
            "inputs": {"input_ids": [1, 32]},
            "outputs": {
                "language_mask": list(pad_mask.shape),
                "language_features": list(feats.shape),
                "language_embeds": list(embeds.shape),
            },
        }
        _export(
            wrapper,
            (dummy_ids,),
            output_dir / "text-encoder.onnx",
            ["input_ids"],
            ["language_mask", "language_features", "language_embeds"],
            {
                "input_ids": {0: "batch"},
                "language_mask": {0: "batch"},
                "language_features": {1: "batch"},
                "language_embeds": {1: "batch"},
            },
            opset,
        )
        del wrapper
        torch.cuda.empty_cache()

    if "vision" in modules:
        wrapper = VisionEncoderWrapper(detector, tracker).float().to(device).eval()
        dummy = torch.randn(1, 3, 1008, 1008, device=device, dtype=torch.float32)
        io = _export(
            wrapper,
            (dummy,),
            output_dir / "vision-encoder.onnx",
            [
                "images",
            ],
            [
                "det_fpn_0",
                "det_fpn_1",
                "det_fpn_2",
                "det_pos_0",
                "det_pos_1",
                "det_pos_2",
                "pvs_high_res_0",
                "pvs_high_res_1",
                "pvs_image_embed",
            ],
            None,
            opset,
        )
        spec["files"]["vision"] = io
        (output_dir / "io_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        del wrapper
        torch.cuda.empty_cache()

    if "grounding" in modules:
        _patch_sam3_for_onnx_export()
        wrapper = GroundingDecoderWrapper(detector).to(device).eval()
        _move_python_tensor_caches(wrapper, device)
        dummy = (
            torch.randn(1, 256, 288, 288, device=device),
            torch.randn(1, 256, 144, 144, device=device),
            torch.randn(1, 256, 72, 72, device=device),
            torch.randn(1, 256, 288, 288, device=device),
            torch.randn(1, 256, 144, 144, device=device),
            torch.randn(1, 256, 72, 72, device=device),
            torch.randn(32, 1, 256, device=device),
            torch.zeros(1, 32, dtype=torch.bool, device=device),
            # geometric prompts are seq-first: N x B x C
            torch.tensor([[[0.5, 0.5, 0.2, 0.2]]], device=device),
            torch.ones(1, 1, dtype=torch.long, device=device),
            torch.zeros(1, 1, dtype=torch.bool, device=device),
            torch.tensor([[[0.5, 0.5]]], device=device),
            torch.ones(1, 1, dtype=torch.long, device=device),
            torch.ones(1, 1, dtype=torch.bool, device=device),
        )
        _export(
            wrapper,
            dummy,
            output_dir / "grounding-decoder.onnx",
            [
                "det_fpn_0",
                "det_fpn_1",
                "det_fpn_2",
                "det_pos_0",
                "det_pos_1",
                "det_pos_2",
                "language_features",
                "language_mask",
                "box_coords",
                "box_labels",
                "box_pad_mask",
                "point_coords",
                "point_labels",
                "point_pad_mask",
            ],
            ["pred_masks", "pred_boxes", "pred_logits", "presence_logits"],
            {
                "det_fpn_0": {0: "batch"},
                "det_fpn_1": {0: "batch"},
                "det_fpn_2": {0: "batch"},
                "det_pos_0": {0: "batch"},
                "det_pos_1": {0: "batch"},
                "det_pos_2": {0: "batch"},
                "language_features": {0: "text_len", 1: "batch"},
                "language_mask": {0: "batch", 1: "text_len"},
                "box_coords": {0: "num_boxes", 1: "batch"},
                "box_labels": {0: "num_boxes", 1: "batch"},
                "box_pad_mask": {0: "batch", 1: "num_boxes"},
                "point_coords": {0: "num_points", 1: "batch"},
                "point_labels": {0: "num_points", 1: "batch"},
                "point_pad_mask": {0: "batch", 1: "num_points"},
                "pred_masks": {0: "batch"},
                "pred_boxes": {0: "batch"},
                "pred_logits": {0: "batch"},
                "presence_logits": {0: "batch"},
            },
            opset,
        )
        spec["files"]["grounding"] = {"file": "grounding-decoder.onnx"}
        del wrapper
        torch.cuda.empty_cache()

    if "prompt" in modules:
        wrapper = PromptDecoderWrapper(tracker).to(device).eval()
        dummy = (
            torch.randn(1, 256, 72, 72, device=device),
            torch.randn(1, 32, 288, 288, device=device),
            torch.randn(1, 64, 144, 144, device=device),
            torch.tensor([[[504.0, 504.0]]], device=device),
            torch.ones(1, 1, dtype=torch.int32, device=device),
            torch.zeros(1, 1, 288, 288, device=device),
            torch.zeros(1, device=device),
        )
        io = _export(
            wrapper,
            dummy,
            output_dir / "prompt-decoder.onnx",
            [
                "image_embed",
                "high_res_0",
                "high_res_1",
                "point_coords",
                "point_labels",
                "mask_input",
                "has_mask_input",
            ],
            ["low_res_masks", "iou_predictions", "object_score_logits"],
            {
                "image_embed": {0: "batch"},
                "high_res_0": {0: "batch"},
                "high_res_1": {0: "batch"},
                "point_coords": {0: "batch", 1: "num_points"},
                "point_labels": {0: "batch", 1: "num_points"},
                "mask_input": {0: "batch"},
                "has_mask_input": {0: "batch"},
                "low_res_masks": {0: "batch", 1: "num_masks"},
                "iou_predictions": {0: "batch", 1: "num_masks"},
                "object_score_logits": {0: "batch"},
            },
            opset,
        )
        spec["files"]["prompt"] = io
        del wrapper

    (output_dir / "io_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"Wrote {output_dir / 'io_spec.json'}")


def main():
    parser = argparse.ArgumentParser(description="Export SAM 3.1 multiplex to ONNX")
    parser.add_argument(
        "--checkpoint",
        default=r"D:\work\dataset\sam3.1\sam3.1_multiplex.pt",
    )
    parser.add_argument(
        "--output-dir",
        default=r"D:\work\dataset\sam3.1\onnx",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--modules",
        nargs="+",
        choices=["vision", "text", "grounding", "prompt"],
        default=None,
    )
    parser.add_argument(
        "--fold-existing",
        action="store_true",
        help="Only fold constant If nodes in already-exported ONNX files (no re-export)",
    )
    parser.add_argument(
        "--convert-fp16",
        action="store_true",
        help="Convert existing ONNX weights to fp16 sidecar *.fp16.onnx for browser WASM",
    )
    parser.add_argument(
        "--fp16-modules",
        nargs="+",
        choices=["vision", "text", "grounding", "prompt"],
        default=None,
        help="Modules to convert with --convert-fp16 (default: vision text)",
    )
    args = parser.parse_args()
    try:
        if args.fold_existing:
            output_dir = Path(args.output_dir)
            folded_any = False
            for onnx_path in sorted(output_dir.glob("*.onnx")):
                if onnx_path.name.endswith(".fp16.onnx") or onnx_path.name.endswith(".onnx.tmp"):
                    continue
                if fold_onnx_file(onnx_path):
                    folded_any = True
            if not folded_any:
                print(f"No constant If nodes found under {output_dir}")
            if not args.convert_fp16:
                return
        if args.convert_fp16:
            output_dir = Path(args.output_dir)
            modules = args.fp16_modules or ["vision", "text"]
            for module in modules:
                source = output_dir / _FP16_FILES[module]
                if not source.exists():
                    raise FileNotFoundError(source)
                convert_onnx_file_to_fp16(source)
            return
        export_all(args)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
