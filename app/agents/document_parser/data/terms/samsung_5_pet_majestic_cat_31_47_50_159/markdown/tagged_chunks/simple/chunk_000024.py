from langchain_core.documents import Document

chunk = Document(
    page_content=('- 아닌 경우에는 본인의 인감증명서, 본인서명사실확인서 또는 안전성과 신뢰성이\n'
 '- 확보된 전자적 수단을 활용한 보험수익자 의사표시의 확인방법 포함)\n'
 '- 4. 기타 보험수익자가 보험금의 수령에 필요하여 제출하는 서류(단, 단체취급 특별약\n'
 '- 관을 부가하는 경우, 사망보험금을 지급할 때 피보험자의 법정상속인이 아닌 자가\n'
 '- 청구하는 경우 법정상속인의 확인서 등)\n'
 '② 제1항 제2호의 사고증명서는 의료법 제3조(의료기관)에서 규정한 국내의 병원이나 의'),
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
 'indexing': {'chunk_id': 'chunk_000024',
              'chunk_char_len': 250,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
