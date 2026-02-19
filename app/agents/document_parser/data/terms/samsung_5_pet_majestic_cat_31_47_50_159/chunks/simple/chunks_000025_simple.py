from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[흥행]\n'
 '영리를 목적으로 연극, 영화, 서커스 등을 요금을 받고 대중에게 보여주는 행위를 말합니다.\n'
 '제6조 (보험금 지급사유의 통지)\n'
 '계약자 또는 피보험자나 보험수익자는 제3조(보험금의 지급사유)에서 정한 보험금 지급 사유의 발생을 안 때에는 지체없이 그 사실을 회사에 '
 '알려야 합니다.\n'
 '제 7조 (보험금의 청구)\n'
 '① 보험수익자는 다음의 서류를 제출하고 보험금을 청구하여야 합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 33},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000025',
              'chunk_char_len': 219,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
