from langchain_core.documents import Document

chunk = Document(
    page_content=('# 제6조(수술의 정의와장소)이 특별약관에 있어서 "수술"이라 함은 동물병원의 수의사 자격을 가진 자(이하 "수\n'
 '의사"라 합니다)에 의하여 치료가 필요하다고 인정된 상해 또는 질병 치료를 위하여\n'
 '수의사법 제17조(개설)에서 규정한 국내의 동물병원에서 수의사의 관리 하에 직접적\n'
 '인 치료를 목적으로 기구를 사용하여 생체에 절단, 절제 등의 조작을 가하는 것을 말\n'
 '합니다. 단, 수술에서 아래에 정한 사항은 제외합니다.\n'
 '1. 흡인(吸引)- 2. 천자(穿刺) 등의 조치\n'
 '- 3. 미용성형 목적의 수술'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
