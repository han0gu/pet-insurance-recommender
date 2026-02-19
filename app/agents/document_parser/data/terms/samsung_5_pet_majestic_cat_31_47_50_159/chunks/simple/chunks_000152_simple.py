from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[이미 발생한 보험금 지급사유에 대한 보험금 지급]\n'
 '계약자, 피보험자 또는 보험수익자가 보험금 청구에 관한 서류를 변조하여 보험금을 청구한 경우, 회사는 그 사실을 안 날부터 1개월 이내에 '
 '계약을 해지할 수 있습니다. 다만, 이 경우에도 회사는 실제 발생한 보험금 지급사유에 대해서는 보험금을 지급합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 44},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000152',
              'chunk_char_len': 177,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
