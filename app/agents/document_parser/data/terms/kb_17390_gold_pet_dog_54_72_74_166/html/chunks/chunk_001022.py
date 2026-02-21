from langchain_core.documents import Document

chunk = Document(
    page_content=('. 상해 또는 질병의 직접적인 치료를 목적으로 "MRI/CT"를 받은 경우 상<br>해<br>2. 백내장 또는 녹내장의 직접적인 치료를 '
 '목적으로 "백내장/녹내장수술"을 받은<br>및<br>경우<br>질<br>3. 이물 섭취 치료를 직접적인 목적으로 치료 중 '
 '"이물제거(내시경)" 또는 "이물<br>병<br>제거(구토유도약물)"를 받은 경우<br>4. 상해로 인한 창상 또는 교상의 직접적인 '
 '치료를 목적으로 "창상/교상치료"를<br>받은 경우<br>반<br>5'),
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
