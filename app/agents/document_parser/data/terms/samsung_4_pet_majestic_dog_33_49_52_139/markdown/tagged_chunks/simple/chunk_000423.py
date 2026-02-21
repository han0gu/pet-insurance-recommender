from langchain_core.documents import Document

chunk = Document(
    page_content=('(Fiberglass Cast)를 병변이 있는 뼈, 관절부위의 둘레 모두에 착용시켜(Circular Cast)\n'
 '감은 다음 굳어지게 하여 치료효과를 가져오는 치료법을 말합니다. 단, 부목(Splint\n'
 'Cast)치료는 제외합니다.② 제1항에서 "부목치료"라 함은 석고붕대 또는 섬유유리붕대(Fiberglass Cast)를 고정 할\n'
 '부분의 일측면 또는 양측면에 착용시키고 대주는 치료법을 말합니다.# 제4조 (특별약관의 소멸)피보험자가 보험기간 중에 사망하였을 경우에는 '
 '"보험료 및 해약환급금 산출방법서"에서'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000423',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
