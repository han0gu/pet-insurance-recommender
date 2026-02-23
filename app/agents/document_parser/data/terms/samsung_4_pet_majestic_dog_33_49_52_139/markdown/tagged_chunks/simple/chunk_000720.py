from langchain_core.documents import Document

chunk = Document(
    page_content=('「보험계약」이라 합니다)을 체결 또는 변경할 때 다음 각 호의 경우 보험계약자(이하\n'
 '「계약자」라 합니다)의 청약과 보험회사(이하「회사」라 합니다)의 승낙으로 계약에\n'
 '부가하여 이루어집니다.- 1. 보험계약을 체결할 때 피보험자의 건강상태가 회사가 정한 기준에 적합하지 않은\n'
 '- 경우\n'
 '- 2. 보험계약을 체결한 후 계약 전 알릴 의무 위반의 효과 등으로 보장을 제한할 경우\n'
 '- ② 제1항에 따라 보장이 제한되는 범위는 의학적으로 인과관계가 있다고 입증된 경우 또'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000720',
              'chunk_char_len': 256,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
