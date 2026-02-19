from langchain_core.documents import Document

chunk = Document(
    page_content=('4. 기타 보험수익자가 보험금의 수령에 필요하여 제출하는 서류\n'
 '② 제1항 제2호의 사고증명서는 수의사법 제12조(진단서 등)에서 규정한 내용에 따라 국 내의 동물병원에서 수의사에 의해 발급한 것이어야 '
 '합니다.\n'
 '【수의사법 제12조(진단서 등)】'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 6},
 'term_type': 'basic',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000034',
              'chunk_char_len': 135,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
