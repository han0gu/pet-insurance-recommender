from langchain_core.documents import Document

chunk = Document(
    page_content=('= [(203만원 - 3만원) × 70%, 200만원] 중 적은 금액\n'
 '= 140만원∙ 예시2- - 피보험자가 부담한 수술당일 의료비 303만원\n'
 '- - 보험금 지급금액\n'
 '- = [(303만원 - 3만원) × 70%, 200만원] 중 적은 금액\n'
 '- = 200만원'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000435',
              'chunk_char_len': 143,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
