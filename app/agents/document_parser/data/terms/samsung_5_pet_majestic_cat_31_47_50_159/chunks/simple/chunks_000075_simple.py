from langchain_core.documents import Document

chunk = Document(
    page_content=('. 회사가 이 계약을 청약할 때 피보험자의 건강상태를 판단할 수 있는 기초자료(건강 진단서 사본 등)에 따라 승낙한 경우에 건강진단서 '
 '사본 등에 명기되어 있는 사항 으로 보험금 지급사유가 발생하였을 때(계약자 또는 피보험자가 회사에 제출한 기 초자료의 내용 중 중요사항을 '
 '고의로 사실과 다르게 작성한 때에는 계약을 해지할 수 있습니다) 5'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 38},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000075',
              'chunk_char_len': 189,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
