from langchain_core.documents import Document

chunk = Document(
    page_content=('- 지났을 때\n'
 '- 4. 회사가 이 계약을 청약할 때 피보험자의 건강상태를 판단할 수 있는 기초자료(건강\n'
 '- 진단서 사본 등)에 따라 승낙한 경우에 건강진단서 사본 등에 명기되어 있는 사항\n'
 '- 으로 보험금 지급사유가 발생하였을 때(계약자 또는 피보험자가 회사에 제출한 기\n'
 '- 초자료의 내용 중 중요사항을 고의로 사실과 다르게 작성한 때에는 계약을 해지할\n'
 '- 수 있습니다)\n'
 '- 5. 보험설계사 등이 계약자 또는 피보험자에게 알릴 기회를 주지 않았거나 계약자 또'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000066',
              'chunk_char_len': 255,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
