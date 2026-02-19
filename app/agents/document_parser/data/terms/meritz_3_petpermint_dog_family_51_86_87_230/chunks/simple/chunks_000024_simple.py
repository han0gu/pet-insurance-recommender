from langchain_core.documents import Document

chunk = Document(
    page_content=('제6조(보험금 지급사유의 통지)\n'
 '계약자 또는 피보험자나 보험수익자는 제3조(보험금의 지급 사유)에서 정한 보험금 지급사유의 발생을 안 때에는 지체 없이 그 사실을 회사에 '
 '알려야 합니다.\n'
 '제7조(보험금의 청구)\n'
 '\uf000 보험수익자는 다음의 서류를 제출하고 보험금을 청구하 여야 합니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 56},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000024',
              'chunk_char_len': 155,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
