from langchain_core.documents import Document

chunk = Document(
    page_content=('. ④ 부활(효력회복)되는 특별약관의 보장개시는 제1항 내지 제3항을 따릅니다. 이 경우 부 활(효력회복)일을 보험계약일로 하여 '
 '제3조(보험금의 지급사유) 제3항을 적용합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 104},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000621',
              'chunk_char_len': 98,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
