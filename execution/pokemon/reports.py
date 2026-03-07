from poke_worlds.execution.vlm import merge_ocr_strings
from poke_worlds.execution.report import ExecutionReport, SimpleReport, EQAReport
from poke_worlds.emulation.pokemon.trackers import PokemonOCRTracker
from typing import Dict, Any


class SimplePokemonExecutionReport(SimpleReport):
    REQUIRED_STATE_TRACKER = PokemonOCRTracker

    def state_info_to_str(self, state_info: Dict[str, Dict[str, Any]]) -> str:
        # just get the OCR text info:
        if "ocr" in state_info and "transition_ocr_texts" in state_info["ocr"]:
            ocr_texts = state_info["ocr"]["transition_ocr_texts"]
            all_ocrs = {}
            for ocr_text_dict in ocr_texts:
                for key in ocr_text_dict:
                    if key not in all_ocrs:
                        all_ocrs[key] = []
                    all_ocrs[key].append(ocr_text_dict[key])
            for key in all_ocrs:
                all_ocrs[key] = merge_ocr_strings(all_ocrs[key])
            return "OCR Results: " + "\n".join(
                [f"{k}: {v}" for k, v in all_ocrs.items()]
            )
        else:
            return ""


class EQAPokemonExecutionReport(EQAReport):
    REQUIRED_STATE_TRACKER = PokemonOCRTracker

    def state_info_to_str(self, state_info):
        return SimplePokemonExecutionReport.state_info_to_str(self, state_info)
