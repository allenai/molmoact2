import inspect
import itertools
import os
import re

from olmo.data.academic_datasets import (
    OkVqa,
    TabWMPDirectAnswer,
    AndroidControl, AI2D, CountBenchQa, RealWorldQa, MathVista, MMMU, ClockBench,
    MuirBench, MantisEval, MMSIBench, MantisInstruct, MMIU, Tulu3SftFiltered, Tulu4Filtered,
    MultiImageVLM3R, Ego3dBench, BLINK, PlotQa, FigureQa, DvQa,
    Omni3D3DOD
)
from olmo.data.academic_datasets_manual import (
    ChartQa, InfoQa, SceneTextQa, DocQa, ScienceQAImageOnly,
    TextVqa, AOkVqa, Vqa2, TallyQa, MulSet, PointBench
)
from olmo.data.debug_pointing_datasets import PointAtTheSquare, PointAtTheSquareVideo, \
    PointAtTheSquareMultiImage
from olmo.data.molmo_hardcodes import Molmo2HardCodes
from olmo.data.lerobot_wrapper import build_lerobot_dataset
from olmo.data.pointing_ab_test import ABPointTestData
from olmo.data.video_datasets import (
    InternVid, Koala, LLaVAVideo178K, MVBench, TempCompass,
    VideoMME, EgoSchema, PerceptionTest, MLVU, LongVideoBench, NeXTQA,
    PeVideo, PlmFGQAEval, PlmFGQATrain, LVBench, LongVideoBenchCaption, Vinoground,
    CLEVRER, STAR, FunQA, TGIF, IntentQA, VideoLocalizedNarratives, RoadTextVQA,
    Paxion, Cinepile, TVQA, VideoEvalProMC, SportsQA, SSV2, ActivityNet, Ego4d,
    COIN, Youcook2, MomentsInTime, NewsVideoQA, SUTDTrafficQA, SocialIQ2, How2QA,
    EpicKitchens, VideoLocalizedNarrativesCaptionHf, QVHighlights, Tomato, TemporalBenchQa,
    MotionBench, Dream1K, MMEVideoOCR, Ego4dCachedClips, VideoHallucer, Countix, Kinetics710,
    CharadesSTA, MotionBenchCaption, LongText,
    CameraBenchTrain, CinepileHf, AcademicTrackingPoints, VSIBench,
)
from olmo.data.vixmo_datasets import (
    VixMoCaptions, VixMoCaptionsQA, VixMoClippedCaption, VixMoClipCaptionsQA,
    VixMoCaptionsEval, VixMoSynClipCaptions, VixMoSynVideoCaptions,
    VixMoSynCaptionsQA, VixMoSynCaptionsSubtitleQA, VixMoPointCountQA, VixMoCaptionsQA2,
    VixMoPoints, VixMoSubtitlePoints, VixMoPointsEval, VixMoHumanQA, Molmo2HumanEval,
    VixMoClipPointing
)
from olmo.data.video_object_tracking_datasets import (
    get_video_object_track_prompt_type,
    Mevis, MevisValid, Burst, RefYoutubeVOS, RefDavis17, LVVIS, YTVIS, ViCaS, ReVOS, VPoS, MoCA,
    ReasonVOS,
    Prolific,
    BboxSinglePointTrack
)

from olmo.data.video_point_tracking_datasets import (
    KubricPointTracking,
    TapDavis, TapKinetics, TapRobotap, TapRGBStacking, DynamicReplica
)

from olmo.data.dataset import Dataset, DATA_HOME
from olmo.data.pixmo_datasets import (
    PixMoDocs, PixMoCount, PixMoPoints, PixMoCapQa, PixMoCap, PixMoPointExplanations,
    PixMoAskModelAnything, PixMoPointsEval, DenseCaptionEval, PixMoClocks,
    CoSyn, CoSynPoint, CorrectionQa, PixMoMultiImageCapQa, PixMoMultiImageMMRCapQa,
    PixMoMultiImageQa, PixMoMultiPoints, CoSynMultiDocs,
    SyntheticGround, GroundCUA
)
from olmo.data.spatial_datasets import (
    SAT, SIMSVSI, VSI590K, RoboPoint, SenseNovaSI, VSTP, RefSpatial, RefSpatialVQA, RefSpatialPoint, CosmosReason1,
    CLEVR, Rel3D, GRiD3D, MindCube
)
from olmo.registry import registry


def get_dataset_by_name(dataset_name, split) -> Dataset:
    vid_home = os.environ["MOLMO_DATA_DIR"]

    # Video Object Tracking Datasets
    # Supported format: {dataset}_{prompt_type}_fps_{fps}_sample_{sampling_type}_{value}
    video_object_track_prompt = get_video_object_track_prompt_type(dataset_name)
    if video_object_track_prompt:
        ''' Extract video object tracking parameters from dataset name'''
        dataset_class, rest = dataset_name.split("_", 1)
        video_object_track_kwargs = {"prompt_type": video_object_track_prompt}
        # fps_match = re.search(r"fps_(\d+)", dataset_name)
        fps_match = re.search(r"(?<!sample_)fps_(\d+)(?:_|$)", dataset_name) # ensure not sample_fps
        sample_fps_match = re.search(r"sample_fps_(\d+)", dataset_name)
        interval_seconds_match = re.search(r"interval_seconds_(\d+\.?\d*)", dataset_name)
        max_objects_match = re.search(r"max_objects_(\d+)", dataset_name)
        if fps_match:
            video_object_track_kwargs["video_fps"] = int(fps_match.group(1))
        if sample_fps_match:
            video_object_track_kwargs["sampling_fps"] = int(sample_fps_match.group(1))
        if interval_seconds_match:
            video_object_track_kwargs["interval_seconds"] = float(interval_seconds_match.group(1))
        if max_objects_match:
            video_object_track_kwargs["max_objects"] = int(max_objects_match.group(1))

        # Extract predicted points if applicable
        predicted_points_match = re.search(r"_predicted_points=(.+\.json)$", dataset_name)
        if predicted_points_match:
            video_object_track_kwargs["predicted_points_file"] = predicted_points_match.group(1)

        if dataset_class == "mevis":
            return Mevis(split, **video_object_track_kwargs)
        elif dataset_class == "mevis-valid":
            return MevisValid(split, **video_object_track_kwargs)
        elif dataset_class == "burst":
            return Burst(split, **video_object_track_kwargs)
        elif dataset_class == "ref-yt-vos":
            return RefYoutubeVOS(split, **video_object_track_kwargs)
        elif dataset_class == "ref-davis17":
            return RefDavis17(split, **video_object_track_kwargs)
        elif dataset_class == "lv-vis":
            return LVVIS(split, **video_object_track_kwargs)
        elif dataset_class == "yt-vis":
            return YTVIS(split, **video_object_track_kwargs)
        elif dataset_class == "vicas":
            return ViCaS(split, **video_object_track_kwargs)
        elif dataset_class == "revos":
            return ReVOS(split, **video_object_track_kwargs)
        elif dataset_class == "vpos":
            return VPoS(split, **video_object_track_kwargs)
        elif dataset_class == "moca":
            return MoCA(split, **video_object_track_kwargs)
        elif dataset_class == "reasonvos":
            return ReasonVOS(split, **video_object_track_kwargs)
        elif dataset_class == "prolific":
            subset_name = rest.split(video_object_track_prompt)[0][:-1] # strp trailing '_'
            return Prolific(split, subset_name, **video_object_track_kwargs)
        elif dataset_class == "bbox-single-point-track":
            subset_name = rest.split(video_object_track_prompt)[0][:-1] # strp trailing '_'
            return BboxSinglePointTrack(split, subset_name, **video_object_track_kwargs)

    # Video Point Tracking Datasets
    if dataset_name == "kubric_random_25_points":
        return KubricPointTracking(split, sample_strategy="random", num_points=25, num_samples_per_video=4)
    elif dataset_name == "kubric_random_5_points_max_frames_60":
        return KubricPointTracking(split, sample_strategy="random", num_points=5, num_samples_per_video=5, max_frames=60)
    elif dataset_name == "kubric_random_5_points_max_frames_30":
        return KubricPointTracking(split, sample_strategy="random", num_points=5, num_samples_per_video=5, max_frames=30)
    elif dataset_name == "tap_davis_5_points_max_frames_60":
        return TapDavis(split, num_points=5, max_frames=60)
    elif dataset_name == "tap_davis_5_points_max_frames_30":
        return TapDavis(split, num_points=5, max_frames=30)

    if dataset_name == "intern_vid":
        return InternVid(split=split)
    if dataset_name == "koala":
        return Koala(split=split)
    if dataset_name == "llava_video_178k_mc":
        return LLaVAVideo178K(split=split, answer_type="multi_choice")
    if dataset_name == "llava_video_178k_mc_split":
        return LLaVAVideo178K(split=split, answer_type="multi_choice", max_per_video=12)
    if dataset_name == "llava_video_178k_mc_flat":
        return LLaVAVideo178K(split=split, answer_type="multi_choice", flat=True)
    if dataset_name == "llava_video_178k_oe":
        return LLaVAVideo178K(split=split, answer_type="open_ended")
    if dataset_name == "llava_video_178k_oe_flat":
        return LLaVAVideo178K(split=split, answer_type="open_ended", flat=True)
    if dataset_name == "llava_video_178k_cap":
        return LLaVAVideo178K(split=split, answer_type="caption")
    if dataset_name == "llava_video_178k_cap_v2":
        return LLaVAVideo178K(split=split, answer_type="caption", partition="default_with_2k_heldout")
    if dataset_name == "llava_video_178k_cap_frame_cap":
        return LLaVAVideo178K(split=split, answer_type="caption", partition="default_with_2k_heldout", include_frame_captions=True)
    if dataset_name == "llava_video_178k_cap_v3":
        return LLaVAVideo178K(split=split, answer_type="caption_no_question", partition="default_with_2k_heldout")
    if dataset_name == "llava_video_178k_cap_v3_dbg":
        return LLaVAVideo178K(split=split, answer_type="none", partition="default_with_2k_heldout")
    if dataset_name == "pe_video":
        return PeVideo(split=split)
    if dataset_name == "llava_video_178k_cap_flat":
        return LLaVAVideo178K(split=split, answer_type="caption", flat=True)
    if dataset_name == "llava_video_178k_cap_no_prompt":
        return LLaVAVideo178K(split=split, answer_type="caption_no_prompt")
    if dataset_name == "llava_video_human_cap":
        return LLaVAVideo178K(split=split, answer_type="caption",
                              id_source=f"{vid_home}/video_captions/video-captions-9k.parquet",
                              cap_source="human")
    if dataset_name == "llava_video_human_cap_id_lv":
        return LLaVAVideo178K(split=split, answer_type="caption",
                              id_source=f"{vid_home}/video_captions/video-captions-9k.parquet",
                              cap_source="lv")
    if dataset_name == "llava_video_oe_academic":
        return LLaVAVideo178K(
                            split,
                            answer_type="open_ended",
                            subset="academic"
                        )
    if dataset_name == "llava_video_mc_academic":
        return LLaVAVideo178K(
                            split,
                            answer_type="multi_choice",
                            subset="academic"
                        )
    if dataset_name == "video_caps_human":
        return VixMoCaptions(split, subset="all", include_video_caption=True)
    if dataset_name == "video_caps_human_baseline":
        return LLaVAVideo178K(split, "caption",
            id_source="/weka/oe-training-default/mm-olmo/video_datasets/video_captions/llava_baseline_caps_36k.parquet",
            cap_source="lv",
            cap_kw="llava_caption"
    )
    if dataset_name == "vixmo_merged_1frame":
        return VixMoCaptions(
            split,
            subset="all",
            n_frame_captions=1,
            include_merged_caption=True,
            include_video_image_merged_caption=True,
        )
    if dataset_name == "molmo2_hardcodes":
        assert split == "train"
        return Molmo2HardCodes()
    if dataset_name == "molmo2_hardcodes_no_video":
        assert split == "train"
        return Molmo2HardCodes(p_video=0)
    if dataset_name == "vixmo_merged_1frame_filtered":
        return VixMoCaptions(
            split,
            subset="all",
            n_frame_captions=1,
            include_merged_caption=True,
            include_video_image_merged_caption=True,
            version="v2_filtered"
        )
    if dataset_name == "vixmo_merged_1frame_v3":
        return VixMoCaptions(
            split,
            subset="all",
            n_frame_captions=1,
            include_merged_caption=True,
            include_video_image_merged_caption=True,
            version="v3"
        )
    if dataset_name == "vixmo_v3":
        return VixMoCaptions(
            split,
            subset="all",
            n_frame_captions=1,
            include_video_caption=True,
            include_merged_caption=True,
            # include_video_image_merged_caption=True,
            version="v3"
        )
    if dataset_name == "vixmo_image_merged":
        return VixMoCaptions(
            split,
            subset="all",
            include_video_image_merged_caption=True,
        )
    if dataset_name == "vixmo3_image_merged":
        return VixMoCaptions(
            split,
            subset="all",
            include_video_image_merged_caption=True,
            version="v3"
        )
    if dataset_name == "vixmo_top_leval_captions":
        return VixMoCaptions(
            split,
            subset="all",
            include_video_transcript=False,
            include_video_caption=True,
            include_merged_caption=True,
            include_video_image_merged_caption=True,
        )
    if dataset_name == "vixmo3_top_level_captions":
        return VixMoCaptions(
            split,
            subset="all",
            include_video_transcript=False,
            include_video_caption=True,
            include_merged_caption=True,
            include_video_image_merged_caption=True,
            version="v3"
        )
    if dataset_name == "vixmo3_top_level_captions_min_3":
        return VixMoCaptions(
            split,
            subset="all",
            include_video_transcript=False,
            include_video_caption=True,
            include_merged_caption=True,
            include_video_image_merged_caption=True,
            min_score=3,
            version="v3"
        )
    if dataset_name == "vixmo3_captions_min_3":
        return VixMoCaptions(
            split,
            subset="all",
            include_video_transcript=False,
            include_video_caption=False,
            include_merged_caption=False,
            include_video_image_merged_caption=True,
            min_score=3,
            version="v3"
        )
    if dataset_name == "vixmo3_short_captions_min_3":
        return VixMoCaptions(
            split,
            subset="all",
            include_video_transcript=False,
            include_video_caption=True,
            include_merged_caption=False,
            include_video_image_merged_caption=False,
            min_score=3,
            version="v3"
        )
    if dataset_name == "vixmo3_top_level_captions_min_3_1frame":
        return VixMoCaptions(
            split,
            subset="all",
            include_video_transcript=False,
            include_video_caption=True,
            include_merged_caption=True,
            include_video_image_merged_caption=True,
            n_frame_captions=1,
            min_score=3,
            version="v3"
        )
    if dataset_name == "vixmo3_top_level_no_merged":
        return VixMoCaptions(
            split,
            subset="all",
            include_video_transcript=False,
            include_video_caption=True,
            include_merged_caption=True,
            include_video_image_merged_caption=False,
            version="v3"
        )
    if dataset_name == "vixmo_clips":
        return VixMoCaptions(
            split,
            subset="all",
            n_clip_captions=4,
            version="v3"
        )

    if dataset_name == "vixmo_top_level_captions0.1":
        return VixMoCaptions(
            split,
            subset="all",
            include_video_transcript=False,
            include_video_caption=True,
            include_merged_caption=True,
            include_video_image_merged_caption=True,
            weight=0.1
        )
    if dataset_name == "vixmo_with_clip_merged":
        return VixMoCaptions(
            split,
            subset="all",
            include_merged_caption=True,
            include_video_image_merged_caption=True,
        )
    if dataset_name == "vixmo_clipped_videos":
        return VixMoClippedCaption(split, subset="all", max_clip_length=6, parts="both",
                                   skip_complete_caption=False)
    if dataset_name == "vixmo_clipped_videos_4":
        return VixMoClippedCaption(split, subset="all", max_clip_length=4, parts="caption",
                                   skip_complete_caption=False)
    if dataset_name == "vixmo_top_leval_captions_2clip":
        return VixMoCaptions(
            split,
            subset="all",
            include_video_caption=True,
            include_merged_caption=True,
            include_video_transcript=True,
            include_video_image_merged_caption=True,
            n_clip_captions=2,
        )
    if dataset_name == "vixmo_top_leval_captions_2frame":
        return VixMoCaptions(
            split,
            subset="all",
            include_video_caption=True,
            include_merged_caption=True,
            include_video_transcript=True,
            include_video_image_merged_caption=True,
            n_frame_captions=2,
            max_caption_per_video=10
        )
    if dataset_name == "video_caps_human_expand_10":
        return VixMoCaptions(
            split,
            subset="all",
            include_video_caption=True,
            expanded_clip_captions=10
        )
    if dataset_name == "video_caps_human_expand_10_merged":
        return VixMoCaptions(
            split,
            include_video_caption=True,
            expanded_clip_captions=10
        )
    if dataset_name == "video_caps_human_expand_10_clip_video":
        return VixMoCaptions(
            split,
            include_video_caption=True,
            expanded_clip_captions=10,
            clip_video=True
        )
    if dataset_name == "video_caps_human_all_video_keywords":
        return VixMoCaptions(
            split,
            include_merged_caption=True,
            include_video_caption=True,
            include_video_transcript=True,
        )
    if dataset_name == "vixmo2_top_level_no_trans":
        return VixMoCaptions(
            split,
            subset="all",
            include_video_caption=True,
            include_merged_caption=True,
            include_video_transcript=False,
            include_video_image_merged_caption=True,
        )
    if dataset_name == "video_caps_oe_random":
        return VixMoCaptionsQA(
            split,
            subset="all",
            answer_type="open_ended",
            format="random"
        )
    if dataset_name == "video_caps_mc_random":
        return VixMoCaptionsQA(
            split,
            subset="all",
            answer_type="multi_choice",
            format="random"
        )
    if dataset_name == "video_caps_mc_sa":
        return VixMoCaptionsQA(
            split,
            subset="all",
            answer_type="multi_choice",
            format="short_answer"
        )
    if dataset_name == "video_caps_oe_sa":
        return VixMoCaptionsQA(
            split,
            subset="all",
            answer_type="open_ended",
            format="short_answer"
        )
    if dataset_name == "video_caps_expand_10_plus_frame_captions":
        return VixMoCaptions(
            split,
            subset="all",
            include_video_image_merged_caption="video_image_merged_caption",
            include_video_caption=True,
            expanded_clip_captions=10,
            n_frame_captions=100,  # all frames captions
        )
    if dataset_name in ["video_caps_human_video_image_merged_caption", "vixmo_image_merged_caption"]:
        return VixMoCaptions(
            split,
            subset="all",
            include_video_image_merged_caption=True,
        )
    if dataset_name in ["vixmo_image_merged_caption_filtered"]:
        return VixMoCaptions(
            split,
            subset="all",
            include_video_image_merged_caption=True,
            version="v2_filtered"
        )

    if dataset_name == "vixmo_syn_clip_caps_filter":
        return VixMoSynClipCaptions(
            split,
            subset="filtered",
        )
    if dataset_name == "vixmo_syn_clip_caps_unfilter":
        return VixMoSynClipCaptions(
            split,
            subset="unfiltered",
        )
    if dataset_name == "vixmo_syn_video_caps":
        return VixMoSynVideoCaptions(
            split,
            subset="all",
        )
    if dataset_name == "vixmo_syn_video_caps_v2":
        return VixMoSynVideoCaptions(
            split,
            subset="all",
            version='v2'
        )
    if dataset_name == "vixmo_syn_video_caps_v3":
        return VixMoSynVideoCaptions(
            split,
            subset="all",
            version='v3'
        )
    if dataset_name == "vixmo_syn_video_capqa":
        return VixMoSynCaptionsQA(
            split,
        )
    if dataset_name == "vixmo_syn_video_capqa_v2":
        return VixMoSynCaptionsQA(
            split,
            version='v2'
        )
    if dataset_name == "vixmo_syn_video_capqa_v2_no_count":
        return VixMoSynCaptionsQA(
            split,
            version='v2',
            exclude_counting=True
        )
    if dataset_name == "vixmo_syn_video_capqa_v3":
        return VixMoSynCaptionsQA(
            split,
            version='v3',
            exclude_counting=True
        )
    if dataset_name == "vixmo_syn_video_capqa_long":
        return VixMoSynCaptionsQA(
            split,
            version='long',
        )
    if dataset_name == "vixmo_syn_video_capqa_long2":
        return VixMoSynCaptionsQA(
            split,
            version='long2',
        )
    if dataset_name == "vixmo_syn_video_capqa_long3":
        return VixMoSynCaptionsQA(
            split,
            version='long3',
        )
    if dataset_name == 'vixmo_syn_video_capqa_with_sub':
        return VixMoSynCaptionsSubtitleQA(split)
    if dataset_name == 'vixmo_syn_video_capqa_with_sub_v2':
        return VixMoSynCaptionsSubtitleQA(split, version='v2')
    if dataset_name == 'vixmo_human_video_capqa_no_count_mc':
        return VixMoCaptionsQA2(split=split, subset='all', answer_type='mc', exclude_counting=True)
    if dataset_name == 'vixmo_human_video_capqa_mc':
        return VixMoCaptionsQA2(split=split, subset='all', answer_type='mc')
    if dataset_name == 'vixmo_human_video_capqa_count_oe':
        return VixMoCaptionsQA2(split=split, subset='count', answer_type='oe')
    if dataset_name == 'vixmo_human_video_capqa_count_mc':
        return VixMoCaptionsQA2(split=split, subset='count', answer_type='mc')
    if dataset_name == 'vixmo_human_video_capqa_motion_oe':
        return VixMoCaptionsQA2(split=split, subset='motion', answer_type='oe')
    if dataset_name == 'vixmo_human_video_capqa_motion_mc':
        return VixMoCaptionsQA2(split=split, subset='motion', answer_type='mc')
    if dataset_name == "vixmo_clip_qa_all":
        return VixMoClipCaptionsQA(split=split, answer_type="all")
    if dataset_name == "vixmo_clip_qa_mc":
        return VixMoClipCaptionsQA(split=split, answer_type="multi_choice")
    if dataset_name == "vixmo_clip_qa_oe":
        return VixMoClipCaptionsQA(split=split, answer_type="open_ended")
    if dataset_name == "vixmo_count_clip_aug":
        return VixMoPointCountQA(split=split, mode='only_count', clip_aug=True)
    if dataset_name == "vixmo_count":
        return VixMoPointCountQA(split=split, mode='only_count')
    if dataset_name == "vixmo_count_mc":
        return VixMoPointCountQA(split=split, mode='mc')
    if dataset_name == "vixmo_count_clip":
        return VixMoPointCountQA(split=split, mode='only_count', use_clip=True)
    if dataset_name == "vixmo_count_clip_mc":
        return VixMoPointCountQA(split=split, mode='mc', use_clip=True)
    if dataset_name == 'vixmo_human_qa':
        return VixMoHumanQA(split=split)
    if dataset_name == 'vixmo_human_qa_flat':
        return VixMoHumanQA(split=split, flat=True)

    if dataset_name == "vixmo_caps_eval":
        return VixMoCaptionsEval(split=split)
    if dataset_name == "vixmo_caps_eval2":
        return VixMoCaptionsEval(split=split, version='v2')

    if dataset_name == "molmo2_human_eval":
        return Molmo2HumanEval(split=split)
    if dataset_name == "molmo2_human_eval_cap_p3":
        return Molmo2HumanEval(split=split, cap_prompt="Describe this video", task="caption")
    if dataset_name == "molmo2_human_eval_cap_p2":
        return Molmo2HumanEval(split=split, cap_prompt="Briefly describe this video", task="caption")
    if dataset_name == "molmo2_human_eval_cap":
        return Molmo2HumanEval(split=split, task="caption")

    if dataset_name == "academic_points":
        return AcademicTrackingPoints(split=split, max_points=60, flat=True)
    if dataset_name == "academic_points_point_then_count":
        return AcademicTrackingPoints(split=split, max_points=60, mode="point_count", flat=True)
    if dataset_name == "academic_points_count":
        return AcademicTrackingPoints(split=split, max_points=60, mode="count", flat=True)

    if dataset_name == "academic_points_count_clip_63s":
        return AcademicTrackingPoints(split=split, max_points=60, max_seconds=63, mode="count", load_clip_times_from_metadata=True)
    if dataset_name == "lvvis_count_clip_63s":
        return AcademicTrackingPoints(split=split, max_points=60, max_seconds=63, mode="count", subset="lvvis", load_clip_times_from_metadata=True)
    if dataset_name == "burst_count_clip_63s":
        return AcademicTrackingPoints(split=split, max_points=60, max_seconds=63, mode="count", subset="burst", load_clip_times_from_metadata=True)

    if dataset_name == "academic_point_train_clip_63s_max100":
        return AcademicTrackingPoints(split=split, max_points=100, max_seconds=63, mode=["point_count", "point"], load_clip_times_from_metadata=True)
    if dataset_name == "academic_point_train_clip_63s":
        return AcademicTrackingPoints(split=split, max_seconds=63, mode=["point_count", "point"], load_clip_times_from_metadata=True)

    if dataset_name.startswith("academic_points_clip_63s"):
        parts = dataset_name.split("_")
        if parts[-1] in ["lvvis", "burst", "ovis", "refdavis17", "mevis", "refyoutube"]:
            fps, subset = parts[-2:]
        else:
            fps, subset = parts[-1], "all"
        fps = float(fps.replace("fps", ""))
        return AcademicTrackingPoints(
            split=split, max_points=60, max_seconds=63, mode=["point_count", "point"],
            subset=subset,
            fake_timestamp_fps=fps,
            load_clip_times_from_metadata=True)
    
    if dataset_name.startswith("academic_points_mf384"):
        return AcademicTrackingPoints(
            split=split,
            max_points=60, 
            max_raw_duration=191,
            max_seconds=191,
            mode=["point_count", "point"],
            load_clip_times_from_metadata=False
        )
    
    if dataset_name == "academic_points_rnd_fps":
        return AcademicTrackingPoints(
            split=split, 
            max_points=60, 
            max_seconds=63, 
            mode="point_count", 
            load_clip_times_from_metadata=True, 
            fake_fps_candidates=[0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0]
        )
    
    if dataset_name == "vixmo_points":
        return VixMoPoints(split=split, max_points=60)
    if dataset_name == "vixmo_ab_test_v2":
        assert split == "test"
        return ABPointTestData()
    if dataset_name == "vixmo_points_subtitle":
        return VixMoSubtitlePoints(split=split, max_points=60)

    if dataset_name == "vixmo_points_point_eval":
        return VixMoPointsEval(split=split)

    if dataset_name == "vixmo_clip_train":
        return VixMoClipPointing(split, mode=("point", "point_count"))
    if dataset_name == "vixmo_clip_counting":
        return VixMoClipPointing(split, mode="point_count")

    if dataset_name == "vixmo_points_count":
        if split in ["val", "test"]:
            # only include "counting" subsets for val/test when evaluating
            return VixMoPoints(split=split, flat=True, max_points=60, mode="count", capability="count")
        return VixMoPoints(split=split, mode="count", max_points=60)

    if dataset_name == "vixmo_points_point_then_count":
        return VixMoPoints(split=split, mode="point_count", max_points=60)
    
    if dataset_name == "vixmo_points_count_then_point":
        return VixMoPoints(split=split, mode="count_point", max_points=60)

    if dataset_name == "vixmo_points_subtitle_point_then_count":
        return VixMoSubtitlePoints(split=split, mode="point_count", max_points=60)

    if dataset_name in "vixmo_points_count_clip_63s":
        if split in ["val", "test"]:
            # only include "counting" subsets for val/test when evaluating
            return VixMoPoints(split=split, flat=True, max_points=60, max_seconds=63, mode="count", capability="count", load_clip_times_from_metadata=True)
        return VixMoPoints(split=split, max_points=60, max_seconds=63, mode="count", load_clip_times_from_metadata=True)

    if dataset_name in "vixmo_points_train_clip_63s_max100":
        return VixMoPoints(split=split, max_points=100, max_seconds=63, mode=["point", "point_count"], load_clip_times_from_metadata=True)

    if dataset_name.startswith("vixmo_points_clip_63s"):
        fps = float(dataset_name.split("_")[-1].replace("fps", ""))
        return VixMoPoints(split=split, max_points=60, max_seconds=63, mode="point_count", fake_timestamp_fps=fps, load_clip_times_from_metadata=True)

    if dataset_name == "vixmo_points_rnd_fps":
        return VixMoPoints(
            split=split, 
            max_points=60, 
            max_seconds=63, 
            mode="point_count", 
            load_clip_times_from_metadata=True, 
            fake_fps_candidates=[0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0]
        )

    if dataset_name.startswith("vixmo_points_minmax"):
        min_points = int(dataset_name.split("_")[3])
        max_points = int(dataset_name.split("_")[4])
        clip = not "no_clip" in dataset_name
        flat = not clip
        count_only = "count_only" in dataset_name
        rnd_fps = "rnd_fps" in dataset_name
        include_unsure = "include_unsure" in dataset_name
        return VixMoPoints(
            split=split, 
            flat=flat,
            min_points=min_points,
            max_points=max_points,
            max_seconds=63 if clip else -1,
            mode="count" if count_only else ["point_count", "point"],
            load_clip_times_from_metadata=clip,
            fake_fps_candidates=[0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0] if rnd_fps else None,
            include_unsure=include_unsure,
            multi_message_short_clips=True
        )

    if dataset_name.startswith("vixmo_points_mf384_minmax"):
        min_points = int(dataset_name.split("_")[4])
        max_points = int(dataset_name.split("_")[5])
        return VixMoPoints(
            split=split,
            min_points=min_points,
            max_points=max_points,
            max_raw_duration=191,
            max_seconds=191,
            multi_message_short_clips=True
        )
    
    if dataset_name == "vixmo_points_objects":
        return VixMoPoints(
            split=split, 
            max_points=60, 
            max_seconds=63, 
            subset="object",
            mode="point_count", 
            fake_timestamp_fps=fps, 
            load_clip_times_from_metadata=True, 
            fake_fps_candidates=[0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0]
        )
    
    if dataset_name == "vixmo_points_count_subtitle_clip_63s":
        return VixMoSubtitlePoints(split=split, max_points=60, max_seconds=63, mode="point_count")

    if dataset_name.startswith("video_eval_pro_mc"):
        parts = dataset_name.split("_")
        if "_use_resize" in dataset_name:
            difficulty = "all" if len(parts) <= 6 else parts[6]
            return VideoEvalProMC(split=split, difficulty=difficulty, use_resize=True)
        difficulty = "all" if len(parts) <= 4 else parts[4]
        return VideoEvalProMC(split=split, difficulty=difficulty)
    if dataset_name == "motionbench_train":
        return MotionBenchCaption(split=split)
    if dataset_name.startswith("motionbench"):
        return MotionBench(split=split)
    if dataset_name == "mvbench_1k":
        return MVBench(split=split, sample=1000)
    if dataset_name.startswith("mvbench"):
        difficulty = dataset_name.split("_")[1] if "_" in dataset_name else "all"
        return MVBench(split=split, difficulty=difficulty)
    if dataset_name.startswith("temp_compass"):
        dataset_name = dataset_name.replace("_disable_api", "")
        task = '_'.join(dataset_name.split("_")[2:]) if len(dataset_name.split("_")) > 2 else "all"
        return TempCompass(split=split, task=task)
    if dataset_name.startswith("video_hallucer"):
        return VideoHallucer(split=split)
    if dataset_name.startswith("plm_fgqa_eval"):
        return PlmFGQAEval(split=split)
    if dataset_name.startswith("plm_fgqa_train"):
        return PlmFGQATrain(split=split)
    if dataset_name == "video_mme_w_subtitle":
        return VideoMME(split=split,  difficulty="all", duration="all", with_subtitle=True)
    if dataset_name.startswith("video_mme"):
        parts = dataset_name.split("_")
        duration = "all" if len(parts) <= 2 else parts[2]
        difficulty = "all" if len(parts) <= 3 else parts[3]
        return VideoMME(split=split, duration=duration, difficulty=difficulty)
    if dataset_name.startswith("perception_test"):
        return PerceptionTest(split=split)
    if dataset_name.startswith("ego_schema"):
        return EgoSchema(split=split)
    if dataset_name.startswith("mlvu_mc"):
        if "use_resize" in dataset_name:
            return MLVU(split=split, task="multiple-choice", use_resize=True)
        return MLVU(split=split, task="multiple-choice")
    if dataset_name == "mlvu_gen":
        return MLVU(split=split, task="generation")
    if dataset_name == "lvbench":
        return LVBench()
    if dataset_name == "long_video_bench_caption":
        return LongVideoBenchCaption(split=split)
    if dataset_name.startswith("long_video_bench_no_subtitle"):
        parts = dataset_name.split("_")
        duration_group = "all" if len(parts) <= 5 else parts[5]
        difficulty = "all" if len(parts) <= 6 else parts[6]
        return LongVideoBench(split=split, allow_subtitle=False, difficulty=difficulty, duration_group=duration_group)
    if dataset_name.startswith("long_video_bench_w_subtitle"):
        return LongVideoBench(split=split, with_subtitle=True)
    if dataset_name.startswith("long_video_bench") and not dataset_name.startswith("long_video_bench_no_subtitle") and not dataset_name.startswith("long_video_bench_caption"):  # Workaround since we don't have elif
        parts = dataset_name.split("_")
        duration_group = "all" if len(parts) <= 3 else parts[3]
        difficulty = "all" if len(parts) <= 4 else parts[4]
        return LongVideoBench(split=split, allow_subtitle=True, difficulty=difficulty, duration_group=duration_group)
    if dataset_name.startswith("nextqa_mc"):
        difficulty = "all" if len(dataset_name.split("_")) == 2 else dataset_name.split("_")[2]
        return NeXTQA(split=split, task="multiple-choice", difficulty=difficulty)
    if dataset_name == "vinoground":
        return Vinoground()
    if dataset_name == "clevrer":
        return CLEVRER(split=split, max_per_video=10)
    if dataset_name == "clevrer_multi_correct":
        return CLEVRER(split=split, max_per_video=10, include_multiple_correct=True)
    if dataset_name == "star":
        return STAR(split=split, max_per_video=10)
    if dataset_name == "star_mc":
        return STAR(split=split, max_per_video=10, answer_type="multi_choice")
    if dataset_name == "funqa":
        return FunQA(split=split, max_per_video=10)
    if dataset_name == "tgif":
        return TGIF(split=split)
    if dataset_name == "intent_qa":
        return IntentQA(split=split)
    if dataset_name == "video_localized_narratives":
        return VideoLocalizedNarratives(split=split)
    if dataset_name == "video_localized_narratives_caption":
        return VideoLocalizedNarrativesCaptionHf(split=split)
    if dataset_name == "road_text_vqa":
        return RoadTextVQA(split=split)
    if dataset_name == "paxion":
        return Paxion(split=split)
    if dataset_name == "cinepile":
        return CinepileHf(split=split, with_subtitle=False)
    if dataset_name == "kinetics":
        return Kinetics710(split=split)
    if dataset_name == "kinetics_qa":
        return Kinetics710(split=split, qa_format=True)
    if dataset_name == "charades_sta":
        return CharadesSTA(split=split)
    if dataset_name == "charades_sta_qa":
        return CharadesSTA(split=split, qa_format=True)
    if dataset_name == "charades_sta_all":
        return CharadesSTA(split=split, task="all")
    if dataset_name == "charades_sta_all_qa":
        return CharadesSTA(split=split, task="all", qa_format=True)
    if dataset_name == "cinepile_with_sub":
        return CinepileHf(split=split, with_subtitle=True)
    if dataset_name == "tvqa":
        return TVQA(split=split)
    if dataset_name == "tvqa_with_sub":
        return TVQA(split=split, with_subtitle=True)
    if dataset_name == "sportsqa_oe":
        return SportsQA(split=split)
    if dataset_name == "camerabench_qa":
        return CameraBenchTrain(split=split)
    if dataset_name == "countix_oe":
        return Countix(split=split, answer_format='oe')
    if dataset_name == "countix_mc":
        return Countix(split=split, answer_format='mc')
    if dataset_name == "activitynet_caption":
        return ActivityNet(split=split, task="captioning")
    if dataset_name == "activitynet_qa":
        return ActivityNet(split=split, task="qa")
    if dataset_name == "activitynet_all":
        return ActivityNet(split=split, task="all")
    if dataset_name == "activitynet_all_qa":
        return ActivityNet(split=split, task="all", qa_format=True)
    if dataset_name == "how2qa":
        return How2QA(split=split)
    if dataset_name == "news_video_qa":
        return NewsVideoQA(split=split)
    if dataset_name == "news_video_qa_filtered":
        return NewsVideoQA(split=split, filter_empty_answers=True)
    if dataset_name == "sutd_trafficqa":
        return SUTDTrafficQA(split=split)
    if dataset_name == "social_iq2":
        return SocialIQ2(split=split)
    if dataset_name == "epic_kitchens":
        return EpicKitchens(split=split)
    if dataset_name == "epic_kitchens_qa":
        return EpicKitchens(split=split, qa_format=True)
    if dataset_name == "ego4d_all":
        return Ego4dCachedClips(split=split, task="all")
    if dataset_name == "ego4d_mq_temporal_grounding":
        return Ego4d(
            split=split,
            task="mq_temporal_grounding",
            video_segment_length=180
        )
    if dataset_name == "ego4d_nlq_temporal_grounding":
        return Ego4d(
            split=split,
            task="nlq_temporal_grounding",
            video_segment_length=180
        )
    if dataset_name == "ego4d_mq_label_start_end":
        return Ego4d(
            split=split,
            task="mq_label_start_end",
            video_segment_length=180
        )
    if dataset_name == "ego4d_mq_label_clip":
        return Ego4d(
            split=split,
            task="mq_label_clip",
            video_segment_length=180
        )
    if dataset_name == "ssv2":
        return SSV2(split=split)
    if dataset_name == "ssv2_qa":
        return SSV2(split=split, qa_format=True)
    if dataset_name == "coin":
        return COIN(split=split)
    if dataset_name == "coin_qa":
        return COIN(split=split, qa_format=True)
    if dataset_name == "coin_all_qa":
        return COIN(split=split, task="all", qa_format=True)
    if dataset_name == "youcook2_caption_clip":
        return Youcook2(split=split, task="caption_clip")
    if dataset_name == "youcook2_caption_start_end":
        return Youcook2(split=split, task="caption_start_end")
    if dataset_name == "youcook2_all":
        return Youcook2(split=split, task="all")
    if dataset_name == "youcook2_qa":
        return Youcook2(split=split, task="caption_clip", qa_format=True)
    if dataset_name == "youcook2_all_qa":
        return Youcook2(split=split, task="all", qa_format=True)
    if dataset_name == "moments_in_time":
        return MomentsInTime(split=split)
    if dataset_name == 'moments_in_time_qa':
        return MomentsInTime(split=split, qa_format=True)
    if dataset_name.startswith("qv_highlights"):  # Allowed -> qv_highlights_min_0.7
        parts = dataset_name.split("_")
        if len(parts) == 4 and "min" == parts[2]:
            return QVHighlights(split=split, minimum=float(parts[3]))
        return QVHighlights(split=split)
    if dataset_name == "tomato":
        return Tomato(split=split)
    if dataset_name == "temporal_bench":
        return TemporalBenchQa(split=split)
    if dataset_name == "temporal_bench_mc":
        return TemporalBenchQa(split=split, format="mc")
    if dataset_name == "dream1k":
        return Dream1K(split="test")
    if dataset_name == "mme_videoocr_mc":
        return MMEVideoOCR(split="test", subset="mc")
    if dataset_name == "mme_videoocr":
        return MMEVideoOCR(split="test", subset="all")

    if dataset_name in ["scifi_document_qa", "pixmo_docs_other"]:
        return PixMoDocs("other", split=split)
    elif dataset_name in ["scifi_table_qa", "pixmo_docs_tables"]:
        return PixMoDocs("tables", split=split)
    elif dataset_name in ["scifi_diagram_qa", "pixmo_docs_diagrams"]:
        return PixMoDocs("diagrams", split=split)
    elif dataset_name in ["scifi_charts_qa", "pixmo_docs_charts"]:
        return PixMoDocs("charts", split=split)

    elif dataset_name in ["pixmo_docs_other_flat"]:
        return PixMoDocs("other", split=split, flat=True)
    elif dataset_name in ["pixmo_docs_charts_flat"]:
        return PixMoDocs("charts", split=split, flat=True)
    elif dataset_name in ["pixmo_docs_tables_flat"]:
        return PixMoDocs("tables", split=split, flat=True)
    elif dataset_name in ["pixmo_docs_diagrams_flat"]:
        return PixMoDocs("diagrams", split=split, flat=True)

    # CoSyn-400K / CoSyn-point
    doc_types = [
        "chart", "chemical", "circuit", "diagram",
        "document", "graphic", "math", "music",
        "nutrition", "table"
    ]
    cosyn_dataset_names = [f"cosyn_{doc_type}{suffix}{flat}" for doc_type, suffix, flat in
                           itertools.product(doc_types, ["", "_exp"], ["", "_flat"])]
    if dataset_name == "cosyn_point":
        return CoSynPoint(split=split)
    elif dataset_name in cosyn_dataset_names:
        doc_type = dataset_name.split("_")[1]
        flat = dataset_name.endswith("_flat")
        return CoSyn(doc_type, split=split, use_exp=dataset_name.endswith("_exp"), flat=flat)
    elif dataset_name.startswith("cosyn_multidoc"):
        parts = dataset_name.split("_")
        if parts[-1] == "exp":
            exp = True
            doc_type = parts[-2]
        else:
            exp = False
            doc_type = parts[-1]
        return CoSynMultiDocs(doc_type, use_exp=exp, split=split, max_images=5)

    # PixMo-Pointing
    elif dataset_name in ["pointing_high_freq", "pixmo_points_high_freq"]:
        return PixMoPoints(kind="high_frequency", split=split, counting=False)
    elif dataset_name in ["point_count_high_freq", "pixmo_points_high_freq_counting"]:
        return PixMoPoints(kind="high_frequency", split=split, counting=True)
    elif dataset_name in ["pointing", "pixmo_points"]:
        return PixMoPoints(kind="basic", split=split, counting=False)
    elif dataset_name in ["point_count", "pixmo_points_counting"]:
        return PixMoPoints(kind="basic", split=split, counting=True)

    elif dataset_name in ["point_at_the_square"]:
        return PointAtTheSquare(split)
    elif dataset_name in ["point_at_the_square_multi"]:
        return PointAtTheSquare(split, min_points=0, max_points=4)
    elif dataset_name in ["point_at_the_square_video_multi"]:
        return PointAtTheSquareVideo(split, min_points=0, max_points=5, min_frames=2, max_frames=8)
    elif dataset_name in ["track_the_square"]:
        return PointAtTheSquareVideo(split, min_points=2, max_points=5, min_frames=4, max_frames=8, tracking=True)
    elif dataset_name in ["point_at_the_square_video_multi_msg"]:
        return PointAtTheSquareVideo(split, min_points=0, max_points=5, min_frames=2, max_frames=8, max_messages=4)
    elif dataset_name in ["point_at_the_square_video_multi_msg_dbg"]:
        return PointAtTheSquareVideo(split, min_points=0, max_points=5, min_frames=2, max_frames=8, max_messages=1)
    elif dataset_name in ["point_at_the_square_video_multi_color"]:
        return PointAtTheSquareVideo(split, min_points=0, max_points=5, min_frames=2, max_frames=8, unique_colors=True)

    elif dataset_name in ["point_at_the_square_multi_image"]:
        return PointAtTheSquareMultiImage(split, min_points=0, max_points=5, max_images=3, min_images=1)

    # More than 60 points will start getting truncated anyway with a seq. len of 2304
    elif dataset_name in ["pixmo_points_train"]:
        return PixMoPoints(kind="basic", split=split, counting="both", max_points=60, max_total_points_per_example=60)
    elif dataset_name in ["pixmo_points_train_flat"]:
        return PixMoPoints(kind="basic", split=split, counting="both", max_points=60, max_total_points_per_example=-1)
    elif dataset_name in ["pixmo_points_flat"]:
        return PixMoPoints(kind="basic", split=split, counting=False, max_points=20, max_total_points_per_example=-1)
    elif dataset_name in ["pixmo_points_high_freq_train"]:
        return PixMoPoints(kind="high_frequency", split=split, counting="both", max_points=60, max_total_points_per_example=60)
    elif dataset_name in ["pixmo_points_train_max200"]:
        return PixMoPoints(kind="basic", split=split, counting="both", max_points=200, max_total_points_per_example=200)
    elif dataset_name in ["pixmo_points_high_freq_train_max200"]:
        return PixMoPoints(kind="high_frequency", split=split, counting="both", max_points=200, max_total_points_per_example=200)
    # PixMoPoints with root_size_factor sampling (e.g., pixmo_points_300000)
    # Note: The number suffix is used as root_size_factor in training mixture,
    # actual sampling is done by the data loader, not here
    elif dataset_name.startswith("pixmo_points_") and dataset_name.split("_")[-1].isdigit():
        return PixMoPoints(kind="basic", split=split, counting=False)
    elif dataset_name in ["pixmo_count_train"]:
        return PixMoCount(split=split, counting="both")

    # PixMo-Point-Explanations
    elif dataset_name in ["point_qa", "pixmo_pointing_explanations"]:
        return PixMoPointExplanations(split=split, split_groups=True)

    # PixMo-Count
    elif dataset_name in ["fast_flickr_count_qa_point_count", "pixmo_count_counting"]:
        return PixMoCount(split=split, counting=True)
    elif dataset_name in ["fast_flickr_count_qa_pointing", "pixmo_count"]:
        return PixMoCount(split=split, counting=False)

    # PixMo-AskModelAnything
    elif dataset_name in ["user_qa", "pixmo_ask_model_anything"]:
        return PixMoAskModelAnything(split=split)
    elif dataset_name in ["pixmo_ask_model_anything_flat"]:
        return PixMoAskModelAnything(split=split, flat=True)
    elif dataset_name in ["pixmo_ask_model_anything_flat_2k"]:
        return PixMoAskModelAnything(split=split, flat=True, sample=2048, skip_counting=True)

    # PixMo-CapQa
    elif dataset_name in ["synthetic_qa_v3", "pixmo_cap_qa"]:
        return PixMoCapQa(split=split)
    elif dataset_name in ["synthetic_qa_v3_as_user_qa", "pixmo_cap_qa_as_user_qa"]:
        return PixMoCapQa(split=split, style="user_qa")

    # PixMo-Cap
    if dataset_name in ["cockatoo_and_transcript_712k_sept6", "pixmo_cap_with_transcripts"]:
        return PixMoCap(split, mode="transcript_and_caption")
    if dataset_name in ["cockatoo_712k_sept6", "pixmo_cap"]:
        return PixMoCap(split, mode="captions")
    if dataset_name in ["pixmo_cap_transcript", "pixmo_transcript"]:
        return PixMoCap(split, mode="transcript")
    # if dataset_name in ["cockatoo_712k_sept6", "pixmo_cap"]:
    #     return PixMoCap(split, mode="captions")
    # if dataset_name in ["pixmo_transcript"]:
    #     return PixMoCap(split, mode="transcript")

    elif dataset_name in ["pixmo_clocks"]:
        return PixMoClocks(split=split)

    if dataset_name == "pointing_eval":
        assert split == "test"
        return PixMoPointsEval(legacy=True)

    if dataset_name == "pointing_eval_v2":
        assert split == "test"
        return PixMoPointsEval()

    # Multi-image Qa
    if dataset_name == "pixmo_multi_image_qa":
        return PixMoMultiImageQa(split=split)
    elif dataset_name == "pixmo_multi_image_qa_multi_only_max5":
        return PixMoMultiImageQa(split=split, multi_image_only=True, max_images=5)
    if dataset_name == "correction_qa":
        return CorrectionQa(split=split)
    elif dataset_name == "correction_qa_multi_only":
        return CorrectionQa(split=split, multi_image_only=True)
    elif dataset_name == "correction_qa_multi_only_max5":
        return CorrectionQa(split=split, multi_image_only=True, max_images=5)
    # Filter out the qa pairs that contain more than 5 images
    elif dataset_name == "correction_qa_train":
        return CorrectionQa(split=split, max_images=5)
    elif dataset_name == "correction_qa_multi_only_train":
        return CorrectionQa(split=split, multi_image_only=True, max_images=5)
    elif dataset_name == "correction_qa_multi_only_6images":
        return CorrectionQa(split=split, multi_image_only=True, max_images=6)

    formats = ["short_answer", "answer_first", "answer_last"]
    for format in formats:
        if dataset_name == f"pixmo_multi_image_cap_qa_{format}":
            return PixMoMultiImageCapQa(split=split, format=format)
    if dataset_name == "pixmo_multi_image_cap_qa":
        return PixMoMultiImageCapQa(split=split, format="all")
    if dataset_name == "pixmo_multi_points":
        return PixMoMultiPoints(split=split)
    if dataset_name == f"pixmo_mmr_multi_image_cap_qa":
        return PixMoMultiImageMMRCapQa(split=split)
    if dataset_name == "pixmo_mmr_multi_image_cap_qa_cot":
        return PixMoMultiImageMMRCapQa(split=split, enable_cot=True)
    # Naming convetion for Mantis-Instruct
    # mantis_instruct_<name>_da_multi_only_flat

    if dataset_name.startswith("mantis_instruct"):
        direct_answer = "_da" in dataset_name
        multi_image_only = "_multi_only" in dataset_name
        flat = "_flat" in dataset_name
        name = dataset_name.replace("_da", "").replace("_flat", "").replace("_multi_only", "")[len("mantis_instruct_"):]
        return MantisInstruct(name, split, direct_answer=direct_answer, multi_image_only=multi_image_only, flat=flat)

    # Academic datasets
    if dataset_name == "android_control":
        return AndroidControl(split)
    if dataset_name == "android_control_ll":
        return AndroidControl(split, mode="ll")
    if dataset_name == "chart_qa":
        return ChartQa(split, weighted=False)
    if dataset_name == "chart_qa_exp":
        return ChartQa(split, weighted=False, use_exp=True)
    if dataset_name == "real_world_qa_no_instruction":
        assert split == "test"
        return RealWorldQa("no_instruction")
    if dataset_name == "chart_qa_weighted":
        return ChartQa(split, weighted=True)
    if dataset_name == "info_qa":
        return InfoQa(split)
    if dataset_name == "doc_qa":
        return DocQa(split)
    if dataset_name == "science_qa_img":
        return ScienceQAImageOnly(split)
    if dataset_name == "coco_2014_vqa_multi":
        return Vqa2(split, multi_question=True)
    if dataset_name == "coco_2014_vqa_8192":
        return Vqa2(split, multi_question=False, sample=8192)
    if dataset_name == "coco_2014_vqa":
        return Vqa2(split, multi_question=False)

    if dataset_name == "text_vqa":
        return TextVqa(split)
    if dataset_name == "plot_qa":
        return PlotQa(split)
    if dataset_name == "figure_qa":
        return FigureQa(dict(train="train", validation="validation1")[split])
    if dataset_name == "dv_qa":
        return DvQa(split)
    if dataset_name == "okvqa":
        return OkVqa(split)
    if dataset_name in ["mmmu"]:
        return MMMU(split)
    if dataset_name in ["mmmu_test"]:
        return MMMU(split)
    if dataset_name in ["mmmu_test_v2"]:
        return MMMU(split, use_multi_image=True)
    if dataset_name == "a_okvqa_da":
        return AOkVqa(split=split, direct_answer=True)
    if dataset_name == "a_okvqa_mc":
        return AOkVqa(split=split, direct_answer=False)
    if dataset_name == "st_qa":
        return SceneTextQa(split=split)
    if dataset_name == "tabwmp_da":
        return TabWMPDirectAnswer(split=split, include_options=False)
    if dataset_name == "countbench_qa":
        assert split == "huggingface"
        return CountBenchQa()
    if dataset_name == "tally_qa":
        return TallyQa(split=split)
    if dataset_name == "ai2_diagram_v2_mix_transparent":
        return AI2D(split=split, boxes="both")
    if dataset_name == "clock_bench":
        return ClockBench(split=split)
    if dataset_name == "tulu3":
        return Tulu3SftFiltered(split=split)
    if dataset_name == "tulu4":
        return Tulu4Filtered(split=split)
    if dataset_name == "tulu4_max_2304":
        return Tulu4Filtered(split=split, max_first_msg_len=2304)
    if dataset_name == "tulu4_max_1024":
        return Tulu4Filtered(split=split, max_first_msg_len=1024)
    if dataset_name == "longtext":
        return LongText(split=split)
    if dataset_name == "longtext_32k":
        return LongText(split=split, seq_len=32000)
    if dataset_name == "longtext_64k":
        return LongText(split=split, seq_len=64000)
    if dataset_name == "dense_caption_eval":
        assert split == "test"
        return DenseCaptionEval()
    elif dataset_name == "math_vista_v2":
        if split == "validation":
            split = "testmini"
        return MathVista(split)
    formats = ["multiple_choice", "short_answer", "answer_first", "answer_last"]
    for format in formats:
        if dataset_name == f"muir_bench_legacy_{format}":
            return MuirBench(split, format=format, legacy=True)
    if dataset_name == "muir_bench":
        return MuirBench(split)
    if dataset_name == "mantis_eval":
        assert split == "test"
        return MantisEval(split)
    elif dataset_name in ["mantis_eval_multi_choice", "mantis_eval_short_answer"]:
        assert split == "test"
        return MantisEval(split, question_type="-".join(dataset_name.split("_")[2:]))
    if dataset_name == "mantis_eval_legacy":
        assert split == "test"
        return MantisEval(split, legacy=True)
    if dataset_name == "point_bench":
        assert split == "test"
        return PointBench()
    elif dataset_name in ["mantis_eval_legacy_multi_choice", "mantis_eval_legacy_short_answer"]:
        assert split == "test"
        return MantisEval(split, question_type="-".join(dataset_name.split("_")[3:]), legacy=True)
    if dataset_name in ["mmsi_bench", "mmsi_bench_legacy_short_answer", "mmsi_bench_legacy_answer_first", "mmsi_bench_legacy_answer_last"]:
        assert split == "test"
        format = "_".join(dataset_name.split("_")[3:]) if dataset_name != "mmsi_bench" else "multiple_choice"
        return MMSIBench(split, format=format)
    if dataset_name == "mmiu":
        assert split == "test"
        return MMIU(split, format="multiple_choice")
    elif dataset_name in ["mmiu_legacy_short_answer", "mmiu_legacy_answer_first", "mmiu_legacy_answer_last"]:
        assert split == "test"
        return MMIU(split, format="_".join(dataset_name.split("_")[2:]), legacy=True)
    if dataset_name == "multi_image_vlm3r":
        return MultiImageVLM3R(split=split)
    elif dataset_name == "multi_image_vlm3r_v2":
        return MultiImageVLM3R(name="vlm3r_v2", split=split)
    elif dataset_name == "blink":
        return BLINK(split)
    elif dataset_name == "mulset":
        assert split == "test"
        return MulSet(split)
    elif dataset_name == "ego3d_bench":
        assert split == "test"
        return Ego3dBench(split)
    elif dataset_name.startswith("vsi_bench"):
        assert split == "test"
        if dataset_name == "vsi_bench":
            return VSIBench(split, multi_image=False)
        elif dataset_name == "vsi_bench_multi_image":
            return VSIBench(split, multi_image=True, max_frames=16)
        else:
            raise NotImplementedError(dataset_name, split)

    # web grounding datasets
    elif dataset_name == "synthetic_ground":
        return SyntheticGround(split=split)
    elif dataset_name == "ground_cua":
        return GroundCUA(split=split)

    elif f"dataset/{dataset_name}" in registry.list():
        return registry.make(f"dataset/{dataset_name}", split=split)
    elif dataset_name.startswith("lerobot:"):
        return build_lerobot_dataset(dataset_name, split)

    # Omni3D 3D Object Detection (VST format, unified FOV 69.16°)
    elif dataset_name == "omni3d_3dod_vst":
        return Omni3D3DOD(split=split)
    elif dataset_name == "omni3d_3dod_vst_50k":
        return Omni3D3DOD(split=split, sample=50000)
    elif dataset_name == "omni3d_3dod_vst_100k":
        return Omni3D3DOD(split=split, sample=100000)

    # Spatial/Embodied Training Datasets
    elif dataset_name == "sat":
        return SAT(split=split, keep_in_memory=True)
    elif dataset_name.startswith("sat_") and dataset_name[4:].isdigit():
        return SAT(split=split, sample=int(dataset_name[4:]), keep_in_memory=True)

    elif dataset_name == "sims_vsi":
        return SIMSVSI(split=split)
    elif dataset_name == "sims_vsi_oe":
        return SIMSVSI(split=split, fmt="oe")
    elif dataset_name == "sims_vsi_mc":
        return SIMSVSI(split=split, fmt="mc")
    elif dataset_name == "sims_vsi_direction":
        return SIMSVSI(split=split, question_category="direction")
    elif dataset_name == "sims_vsi_distance":
        return SIMSVSI(split=split, question_category="distance")
    elif dataset_name == "sims_vsi_counting":
        return SIMSVSI(split=split, question_category="counting")
    elif dataset_name == "sims_vsi_other":
        return SIMSVSI(split=split, question_category="other")
    elif dataset_name.startswith("sims_vsi_") and dataset_name.split("_")[-1].isdigit():
        return SIMSVSI(split=split, sample=int(dataset_name.split("_")[-1]))

    elif dataset_name == "vsi_590k":
        return VSI590K(split=split)
    elif dataset_name == "vsi_590k_video":
        return VSI590K(split=split, media_type="video")
    elif dataset_name == "vsi_590k_image":
        return VSI590K(split=split, media_type="image")
    elif dataset_name.startswith("vsi_590k_video_") and dataset_name.split("_")[-1].isdigit():
        return VSI590K(split=split, media_type="video", sample=int(dataset_name.split("_")[-1]))
    elif dataset_name.startswith("vsi_590k_image_") and dataset_name.split("_")[-1].isdigit():
        return VSI590K(split=split, media_type="image", sample=int(dataset_name.split("_")[-1]))
    elif dataset_name.startswith("vsi_590k_") and dataset_name.split("_")[-1].isdigit():
        return VSI590K(split=split, sample=int(dataset_name.split("_")[-1]))

    elif dataset_name == "robopoint":
        return RoboPoint(split=split)
    elif dataset_name == "robopoint_pointing":
        return RoboPoint(split=split, task_type="pointing")
    elif dataset_name == "robopoint_detection":
        return RoboPoint(split=split, task_type="detection")
    elif dataset_name == "robopoint_qa":
        return RoboPoint(split=split, task_type="qa")
    # More specific patterns MUST come before generic robopoint_ pattern
    elif dataset_name.startswith("robopoint_pointing_") and dataset_name.split("_")[-1].isdigit():
        return RoboPoint(split=split, task_type="pointing", sample=int(dataset_name.split("_")[-1]))
    elif dataset_name.startswith("robopoint_detection_") and dataset_name.split("_")[-1].isdigit():
        return RoboPoint(split=split, task_type="detection", sample=int(dataset_name.split("_")[-1]))
    elif dataset_name.startswith("robopoint_qa_") and dataset_name.split("_")[-1].isdigit():
        return RoboPoint(split=split, task_type="qa", sample=int(dataset_name.split("_")[-1]))
    # Generic pattern must come last
    elif dataset_name.startswith("robopoint_") and dataset_name.split("_")[-1].isdigit():
        return RoboPoint(split=split, sample=int(dataset_name.split("_")[-1]))

    elif dataset_name == "sensenova_si":
        return SenseNovaSI(split=split)
    elif dataset_name.startswith("sensenova_si_") and dataset_name.split("_")[-1].isdigit():
        return SenseNovaSI(split=split, sample=int(dataset_name.split("_")[-1]))

    elif dataset_name == "vst_p":
        return VSTP(split=split)
    elif dataset_name == "vst_p_single":
        return VSTP(split=split, image_type="single")
    elif dataset_name == "vst_p_multi":
        return VSTP(split=split, image_type="multi")
    elif dataset_name.startswith("vst_p_single_") and dataset_name.split("_")[-1].isdigit():
        return VSTP(split=split, image_type="single", sample=int(dataset_name.split("_")[-1]))
    elif dataset_name.startswith("vst_p_multi_") and dataset_name.split("_")[-1].isdigit():
        return VSTP(split=split, image_type="multi", sample=int(dataset_name.split("_")[-1]))
    elif dataset_name.startswith("vst_p_") and dataset_name.split("_")[-1].isdigit():
        return VSTP(split=split, sample=int(dataset_name.split("_")[-1]))

    elif dataset_name == "refspatial":
        return RefSpatial(split=split)
    elif dataset_name == "refspatial_choice":
        return RefSpatial(split=split, qa_type="choice")
    elif dataset_name == "refspatial_reasoning":
        return RefSpatial(split=split, qa_type="reasoning")
    # More specific patterns MUST come before generic refspatial_ pattern
    elif dataset_name.startswith("refspatial_choice_") and dataset_name.split("_")[-1].isdigit():
        return RefSpatial(split=split, qa_type="choice", sample=int(dataset_name.split("_")[-1]))
    elif dataset_name.startswith("refspatial_reasoning_") and dataset_name.split("_")[-1].isdigit():
        return RefSpatial(split=split, qa_type="reasoning", sample=int(dataset_name.split("_")[-1]))

    # RefSpatialVQA: Non-coordinate spatial questions (yes/no, left/right, etc.)
    elif dataset_name == "refspatial_vqa":
        return RefSpatialVQA(split=split)
    elif dataset_name.startswith("refspatial_vqa_") and dataset_name.split("_")[-1].isdigit():
        return RefSpatialVQA(split=split, sample=int(dataset_name.split("_")[-1]))

    # RefSpatialPoint: Coordinate questions converted to Molmo2 pointing format
    elif dataset_name == "refspatial_point":
        return RefSpatialPoint(split=split)
    elif dataset_name.startswith("refspatial_point_") and dataset_name.split("_")[-1].isdigit():
        return RefSpatialPoint(split=split, sample=int(dataset_name.split("_")[-1]))

    # Generic pattern must come last
    elif dataset_name.startswith("refspatial_") and dataset_name.split("_")[-1].isdigit():
        return RefSpatial(split=split, sample=int(dataset_name.split("_")[-1]))

    # Cosmos-Reason1 dataset (embodied reasoning)
    elif dataset_name == "cosmos_reason1":
        return CosmosReason1(split=split)
    elif dataset_name == "cosmos_reason1_human":
        # Human annotations only (filters out VLM-generated verbose answers)
        return CosmosReason1(split=split, use_human_annotations=True)
    elif dataset_name == "cosmos_reason1_understanding":
        return CosmosReason1(split=split, task_type="understanding")
    elif dataset_name == "cosmos_reason1_reasoning":
        return CosmosReason1(split=split, task_type="reasoning")
    elif dataset_name == "cosmos_reason1_robovqa":
        return CosmosReason1(split=split, subset="robovqa")
    # RoboVQA with human annotations (shorthand)
    elif dataset_name == "robovqa":
        return CosmosReason1(split=split, subset="robovqa", use_human_annotations=True)
    elif dataset_name.startswith("robovqa_") and dataset_name.split("_")[-1].isdigit():
        return CosmosReason1(split=split, subset="robovqa", use_human_annotations=True, sample=int(dataset_name.split("_")[-1]))
    elif dataset_name == "cosmos_reason1_bridgev2":
        return CosmosReason1(split=split, subset="bridgev2")
    elif dataset_name == "cosmos_reason1_agibot":
        return CosmosReason1(split=split, subset="agibot")
    elif dataset_name == "cosmos_reason1_holoassist":
        return CosmosReason1(split=split, subset="holoassist")
    elif dataset_name.startswith("cosmos_reason1_human_") and dataset_name.split("_")[-1].isdigit():
        return CosmosReason1(split=split, use_human_annotations=True, sample=int(dataset_name.split("_")[-1]))
    elif dataset_name.startswith("cosmos_reason1_") and dataset_name.split("_")[-1].isdigit():
        return CosmosReason1(split=split, sample=int(dataset_name.split("_")[-1]))

    # ============================================================================
    # Abstract/Commonsense Reasoning Datasets (V3)
    # ============================================================================

    # CLEVR: Compositional Language and Elementary Visual Reasoning
    elif dataset_name == "clevr":
        return CLEVR(split=split)
    elif dataset_name.startswith("clevr_") and dataset_name.split("_")[-1].isdigit():
        return CLEVR(split=split, sample=int(dataset_name.split("_")[-1]))

    # Rel3D: Spatial Relation Classification in 3D Scenes
    elif dataset_name == "rel3d":
        return Rel3D(split=split)
    elif dataset_name.startswith("rel3d_") and dataset_name.split("_")[-1].isdigit():
        return Rel3D(split=split, sample=int(dataset_name.split("_")[-1]))

    # GRiD-3D: Grounding Relative Directions in 3D Scenes
    elif dataset_name == "grid3d":
        return GRiD3D(split=split)
    elif dataset_name.startswith("grid3d_") and dataset_name.split("_")[-1].isdigit():
        return GRiD3D(split=split, sample=int(dataset_name.split("_")[-1]))

    # MindCube: Spatial Mental Modeling from Limited Views
    elif dataset_name == "mindcube":
        return MindCube(split=split)
    elif dataset_name.startswith("mindcube_") and dataset_name.split("_")[-1].isdigit():
        return MindCube(split=split, sample=int(dataset_name.split("_")[-1]))

    raise NotImplementedError(dataset_name, split)


_MISSING_INIT = object()


def get_all_dataset_classes():
    """Return dataset classes imported into this module."""
    return [
        value
        for value in globals().values()
        if inspect.isclass(value) and issubclass(value, Dataset)
    ]


def get_dataset_class_by_name(dataset_name: str):
    """Resolve a configured dataset name without initializing local data.

    ``get_dataset_by_name`` is also the canonical mapping from mixture aliases to
    concrete classes, but most dataset constructors immediately read their local
    processed copy. Temporarily replacing constructors lets the download CLI use
    that mapping and call the class-level ``download`` method first.
    """
    if dataset_name in {
        "lvbench",
        "mmiu",
        "point_bench",
        "real_world_qa_no_instruction",
        "pixmo_point_eval2",
    }:
        split = "test"
    elif dataset_name == "countbench_qa":
        split = "huggingface"
    else:
        split = "train"

    dataset_classes = get_all_dataset_classes()
    original_initializers = {
        cls: cls.__dict__.get("__init__", _MISSING_INIT) for cls in dataset_classes
    }
    try:
        for cls in dataset_classes:
            cls.__init__ = lambda *args, **kwargs: None
        return type(get_dataset_by_name(dataset_name, split))
    finally:
        for cls, initializer in original_initializers.items():
            if initializer is _MISSING_INIT:
                del cls.__init__
            else:
                cls.__init__ = initializer


def download_dataset_by_name(dataset_name: str, n_procs: int = 8) -> None:
    """Download a configured dataset without requiring a pre-existing local copy."""
    dataset_class = get_dataset_class_by_name(dataset_name)
    dataset_class.download(n_procs=n_procs)
