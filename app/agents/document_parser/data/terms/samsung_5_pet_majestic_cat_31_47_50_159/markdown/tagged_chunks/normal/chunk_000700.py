from langchain_core.documents import Document

chunk = Document(
    page_content=('<유의사항>\n'
 '회사는 제2조(특별면책조건의 내용) 제1항 각 호의 질병을 직접적인 원인으로 보험료 납입면제 사\n'
 '유가 발생한 경우 보험료 납입을 면제하여 드리지 않습니다.제3조 (보험료의 납입을 연체하여 해지된 특별약관의 부활(효력회복))회사는 이 '
 '특별약관의 부활(효력회복) 청약을 받은 경우에는 계약의 부활(효력회복)을 승'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000700',
              'chunk_char_len': 179,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
