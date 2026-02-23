from langchain_core.documents import Document

chunk = Document(
    page_content=('우 보험료 납입을 면제하여 드리지 않습니다.제3조 (보험료의 납입을 연체하여 해지된 특별약관의 부활(효력회복))회사는 이 특별약관의 '
 '부활(효력회복)청약을 받은 경우에는 계약의 부활(효력회복)을 승\n'
 '낙한 경우에 한하여 보험계약「보험료의 납입을 연체하여 해지된 특별약관의 부활(효력'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000673',
              'chunk_char_len': 155,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
