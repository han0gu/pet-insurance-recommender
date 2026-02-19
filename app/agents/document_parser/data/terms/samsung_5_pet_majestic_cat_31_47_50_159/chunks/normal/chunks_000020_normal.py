from langchain_core.documents import Document

chunk = Document(
    page_content=('제 5조 (보험금을 지급하지 않는 사유)\n'
 '① 회사는 다음 중 어느 한 가지로 제3조(보험금의 지급사유)에서 정한 보험금 지급사유 가 발생한 때에는 보험금을 지급하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 32},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000020',
              'chunk_char_len': 97,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
