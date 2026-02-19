from langchain_core.documents import Document

chunk = Document(
    page_content=('확보된 전자적 수단을 활용한 보험수익자 의사표시의 확인방법 포함)\n'
 '5. 기타 보험수익자가 보험금의 수령에 필요하여 제출하는 서류\n'
 '② 제1항 제3호의 사고증명서는 의료법 제3조(의료기관)에 규정한 국내의 병원이나 의원 또는 국외의 의료관련법에서 정한 의료기관에서 '
 '발급한 것이어야 합니다.\n'
 '제 5조 (특별약관의 소멸)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 89},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000486',
              'chunk_char_len': 175,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
