from langchain_core.documents import Document

chunk = Document(
    page_content=('② 계약자 또는 피보험자가 제1항 각호의 통지를 게을리하여 손해가 증가된 때에는 회사 는 그 증가된 손해를 보상하여 드리지 않으며, '
 '제1항제3호의 통지를 게을리 한 때에는 소송비용과 변호사비용도 보상하여 드리지 않습니다. 다만, 계약자 또는 피보험자가 상 법 제657조 '
 '제1항에 의해 보험사고의 발생을 회사에 알린 경우에는 제3조(보상하는 손 해) 제3항 제1호 및 제2호 ‘다’목 또는 ‘라’목의 비용에 '
 '대하여 보상한도액을 한도로 보 상하여 드립니다.\n'
 '제6조(보험금의 청구)'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 24},
 'term_type': 'special',
 'clause': {'clause_type': 'limit', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000148',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
