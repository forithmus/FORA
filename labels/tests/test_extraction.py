"""Three-pass engines (CT training, CT scoring, X-ray) with a fake inferer; prompt files format without errors."""
import json

import pytest

from labels.ct.three_pass import ThreePass, parse_json, repair_truncated
from labels.cxr.extract_findings import CXRThreePass
from labels.llm import load_config

CT_CONFIGS = ["configs/ct_extract_global.json", "configs/ct_extract_per_protocol.json",
              "configs/ct_extract_global_retry2.json", "report_scoring/configs/ct_scoring_extract.json"]
P1 = json.dumps({"asserted": [{"leaf": "Pneumonia", "quote": "q"}], "denied": [], "uncertain": [],
                 "other": [{"finding": "x", "quote": "y"}]})
P3 = json.dumps({"final_asserted": [{"leaf": "Pleural effusion", "quote": "q"}], "final_denied": [], "final_uncertain": []})


def answer_for(p3_reply):
    def answer(msgs):
        s = msgs[0]["content"]
        if "FINAL" in s:
            return p3_reply
        if "quality-control" in s:
            return '{"verdict":"PASS","rejected":[],"repolarized":[]}'
        return P1
    return answer


@pytest.mark.parametrize("cfg", CT_CONFIGS)
def test_ct_three_pass(cfg, fake_inferer):
    e = ThreePass(load_config(cfg))
    mod = e.protocols.load(7)
    inf = fake_inferer(answer_for(P3))
    r = e.process_modality(inf, mod, [(0, "Findings text " * 10, "")])
    assert r[0]["verdict"] == "pass" and r[0]["asserted"][0]["leaf"] == "Pleural effusion" and r[0]["other"][0]["finding"] == "x"
    sys1 = inf.calls[0][0][0][0]["content"]
    assert mod["title"] in sys1 and "{vocabulary}" not in sys1 and mod["leaves"][0]["name"] in sys1
    r = e.process_modality(fake_inferer(answer_for("garbage")), mod, [(0, "f", "")])
    assert r[0]["verdict"] == "p3_unparsed" and r[0]["asserted"][0]["leaf"] == "Pneumonia"


def test_repair():
    cut = '{"final_asserted":[{"leaf":"A","quote":"a"},{"leaf":"B","quote":"b"}],"final_denied":[{"leaf":"C","qu'
    assert parse_json(cut) is None
    fixed = repair_truncated(cut)
    assert [x["leaf"] for x in fixed["final_asserted"]] == ["A", "B"] and "final_denied" not in fixed
    e = ThreePass(load_config("configs/ct_extract_global_retry2.json"))
    assert e.resolve(P1, cut)["verdict"] == "p3_repaired"
    assert ThreePass(load_config("configs/ct_extract_global.json")).resolve(P1, cut)["verdict"] == "p3_unparsed"


def test_cxr_three_pass(fake_inferer):
    e = CXRThreePass(load_config("configs/cxr_extract.json"))
    assert len(e.voc["leaves"]) == 304
    r = e.process(fake_inferer(answer_for(P3)), [(0, "Heart size normal.", "")])
    assert r[0]["verdict"] == "pass"
