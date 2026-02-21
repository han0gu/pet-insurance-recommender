from langchain_core.documents import Document

chunk = Document(
    page_content=('- 이 지났을 때\n'
 '- 4. 회사가 이 계약을 청약할 때 반려묘의 건강상태를 판단할 수 있는 기초자료(건강진\n'
 '- 단서 사본 등)에 따라 승낙한 경우에 건강진단서 사본 등에 명기되어 있는 사항으\n'
 '- 101 -로 보험금 지급사유가 발생하였을 때(계약자 또는 피보험자가 회사에 제출한 기초\n'
 '자료의 내용 중 중요사항을 고의로 사실과 다르게 작성한 때에는 이 특별약관을\n'
 '해지할 수 있습니다)5. 보험설계사 등이 계약자 또는 피보험자에게 알릴 기회를 주지 않았거나 계약자 또'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000495',
              'chunk_char_len': 258,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
