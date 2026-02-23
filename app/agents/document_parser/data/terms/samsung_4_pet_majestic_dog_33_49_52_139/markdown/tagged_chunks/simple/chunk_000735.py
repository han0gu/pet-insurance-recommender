from langchain_core.documents import Document

chunk = Document(
    page_content=('- ⑨ 제1항의 특정신체부위와 특정질병은 4개 이내에서 선택하여 부가할 수 있습니다.\n'
 '<유의사항>회사는 제2조(특별면책조건의 내용) 제1항 각 호의 질병을 직접적인 원인으로 보험료 납입면제 사\n'
 '유가 발생한 경우 보험료 납입을 면제하여 드리지 않습니다.# 제3조 (보험료의 납입을 연체하여 해지된 특별약관의 부활(효력회복))회사는 '
 '이 특별약관의 부활(효력회복) 청약을 받은 경우에는 계약의 부활(효력회복)을 승'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000735',
              'chunk_char_len': 228,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
