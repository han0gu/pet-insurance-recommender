from langchain_core.documents import Document

chunk = Document(
    page_content=('【예시】 보험금 지급사유가 2022년 1월 1일에 발생하였음에도 2025년 1월 1일까지 보험금을 청구하지 않는 경우 소멸시효가 완성되어 '
 '보험금을 지급받지 못 할 수 있습니다. 다만, 2025년 1월 1일이 토요일 또는 공휴일 일 경우 그 다음 첫 영업일에 소멸시효가 '
 '완성됩니다.\n'
 '제34조(약관의 해석)\n'
 '① 회사는 신의성실의 원칙에 따라 공정하게 약관을 해석하여야 하며 계약자에 따라 다르게 해석하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 19},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000100',
              'chunk_char_len': 229,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
