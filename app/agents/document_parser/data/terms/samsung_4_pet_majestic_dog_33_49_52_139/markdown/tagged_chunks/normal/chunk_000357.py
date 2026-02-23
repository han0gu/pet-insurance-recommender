from langchain_core.documents import Document

chunk = Document(
    page_content=('- 다.\n'
 '- ③ 제1항에서 정한 상해흉터복원(성형) 수술비는 하나의 사고에 대하여 500만원을 한도\n'
 '- 로 지급합니다. 다만, 동일부위에 대한 성형수술을 2회 이상 받은 경우에는 최초로 받\n'
 '- 은 수술에 대해서만 지급합니다.\n'
 '<용어풀이>[안면부, 상지, 하지]- 1. 안면부란 이마를 포함하여 목까지의 얼굴부분을 말합니다.\n'
 '- 2. 상지란 어깨관절 이하의 팔과 손가락 부분을 말합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000357',
              'chunk_char_len': 215,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
