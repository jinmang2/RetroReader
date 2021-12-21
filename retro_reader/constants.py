import os
from datasets import Sequence, Value, Features
from datasets import Dataset, DatasetDict


EXAMPLE_FEATURES = Features(
    {
        "guid": Value(dtype="string", id=None),
        "question": Value(dtype="string", id=None),
        "context": Value(dtype="string", id=None),
        "answers": Sequence(
            feature={
                "text": Value(dtype="string", id=None),
                "answer_start": Value(dtype="int32", id=None),
            },
        ),
        "is_impossible": Value(dtype="bool", id=None),
        "title": Value(dtype="string", id=None),
        "classtype": Value(dtype="string", id=None),
        "source": Value(dtype="string", id=None),
        "dataset": Value(dtype="string", id=None),
    }
)

SKETCH_TRAIN_FEATURES = Features(
    {
        "input_ids": Sequence(feature=Value(dtype='int32', id=None)),
        "attention_mask": Sequence(feature=Value(dtype='int8', id=None)),
        "token_type_ids": Sequence(feature=Value(dtype='int8', id=None)),
        "labels": Value(dtype='int64', id=None),
    }
)

SKETCH_EVAL_FEATURES = Features(
    {
        "input_ids": Sequence(feature=Value(dtype='int32', id=None)),
        "attention_mask": Sequence(feature=Value(dtype='int8', id=None)),
        "token_type_ids": Sequence(feature=Value(dtype='int8', id=None)),
        "labels": Value(dtype='int64', id=None),
        "example_id": Value(dtype='string', id=None),
    }
)

INTENSIVE_TRAIN_FEATUERS = Features(
    {
        "input_ids": Sequence(feature=Value(dtype='int32', id=None)),
        "attention_mask": Sequence(feature=Value(dtype='int8', id=None)),
        "token_type_ids": Sequence(feature=Value(dtype='int8', id=None)),
        "start_positions": Value(dtype='int64', id=None),
        "end_positions": Value(dtype='int64', id=None),
        "is_impossibles": Value(dtype='float64', id=None),
    }
)

INTENSIVE_EVAL_FEATUERS = Features(
    {
        "input_ids": Sequence(feature=Value(dtype='int32', id=None)),
        "attention_mask": Sequence(feature=Value(dtype='int8', id=None)),
        "token_type_ids": Sequence(feature=Value(dtype='int8', id=None)),
        "offset_mapping": Sequence(
            feature=Sequence(
                feature=Value(dtype='int64', id=None)
            )
        ),
        "example_id": Value(dtype='string', id=None),
    }
)

QUESTION_COLUMN_NAME = "question"
CONTEXT_COLUMN_NAME = "context"
ANSWER_COLUMN_NAME = "answers"
ANSWERABLE_COLUMN_NAME = "is_impossible"
ID_COLUMN_NAME = "guid"

SCORE_EXT_FILE_NAME = "cls_score.json"
INTENSIVE_PRED_FILE_NAME = "predictions.json"
NBEST_PRED_FILE_NAME = "nbest_predictions.json"
SCORE_DIFF_FILE_NAME = "null_odds.json"

DEFAULT_CONFIG_FILE = os.path.join(
    os.path.realpath(__file__), "args/default_config.yaml"
)

KO_QUERY_HELP_TEXT = "질문을 입력해주세요!"
KO_CONTEXT_HELP_TEXT = "문맥을 입력해주세요!"

EN_QUERY_HELP_TEXT = "Plz enter your question!"
EN_CONTEXT_HELP_TEXT = "Plz enter your context!"

KO_EXAMPLE_QUERY = "이순신은 어느 시대의 무신이야?"
KO_EXAMPLE_CONTEXTS = """
16세기 조선의 무신으로, 일본이 조선을 침공하여 일어난 전쟁인 임진왜란 당시 조선 수군을 통솔했던 제독이자 구국영웅이다.
            
침략군과 교전하여 천재적인 활약상을 펼치고 중앙 지원 없이 자급자족을 해낸 군 지휘관이자, 휘하 인사들에게 법에 따른 원칙을 요구하면서도 뚜렷한 성공률과 부족함 없는 처우를 보장한 상관, 지방관 시절 백성들에게 선정을 베풀고 전시에도 그들을 위무하고 구제한 목민관, 고위 관료와 접선 및 축재를 거부하고 공정과 국익, 절제를 중시한 인격자, 자신이 관할한 지역의 백성과 병사에게 각종 사업을 장려하여 많은 수효를 얻어낸 행정가, 그리고 왕을 위시한 조정의 핍박으로 사형수가 되거나 후임자의 실책으로 군사·군선들을 거의 상실하거나 어머니와 아들을 잃는 등 많은 수난을 겪고도 명량 해전 등에 임하며 굴하지 않은 철인의 면모까지 갖춰 조선 중기의 명장을 넘어 한국사 최고 위인의 반열까지 오른 인물이다.

생전부터 그를 사적으로 알고 있던 인근 백성이나 군졸, 일부 장수와 재상들로부터 뛰어난 인물로 평가받았고 그렇지 않더라도 명성이 제법 있었으며 전사 소식에 많은 이가 남녀노소를 불문하고 크게 슬퍼했다고 전해진다. 사후 조정은 관직을 추증했고 선비들은 찬양시(詩)를 지었으며 백성들은 추모비를 세우는 등, 이순신은 오래도록 많은 추앙을 받아왔다. 이는 일제강점기를 거쳐 현대에도 마찬가지로, 이순신은 대한민국 국민들이 가장 존경하는 위인 중 한 명으로 꼽히며 현대 한국에서 성웅이라는 최상급 수사가 이름 앞에 붙어도 어떤 이의도 제기받지 않는, 세종과 함께 한국인에게 가장 사랑받는 한국사 양대 위인이다. 가장 존경하는 위인을 묻는 설문조사에서도 세종대왕과 1, 2위를 다투며 충무공이라는 시호도 실제로는 김시민과 같은 여러 장수들이 받은 시호이지만 현대 한국인들은 이순신 전용 시호로 인식한다.
""".strip()

EN_EXAMPLE_QUERY = "When did Beyonce start becoming popular?"
EN_EXAMPLE_CONTEXTS = """
Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny\'s Child. Managed by her father, Mathew Knowles, the group became one of the world\'s best-selling girl groups of all time. Their hiatus saw the release of Beyoncé\'s debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".
""".strip()