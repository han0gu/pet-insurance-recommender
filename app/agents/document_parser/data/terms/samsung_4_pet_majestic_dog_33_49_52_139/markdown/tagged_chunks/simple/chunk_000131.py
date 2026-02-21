from langchain_core.documents import Document

chunk = Document(
    page_content=('- 지급사유를 발생시킨 경우\n'
 '- 2. 계약자, 피보험자 또는 보험수익자가 보험금 청구에 관한 서류에 고의로 사실과 다\n'
 '- 른 것을 기재하였거나 그 서류 또는 증거를 위조 또는 변조한 경우. 다만, 이미 보\n'
 '- 험금 지급사유가 발생한 경우에는 보험금 지급에 영향을 미치지 않습니다.\n'
 '<용어풀이>[이미 발생한 보험금 지급사유에 대한 보험금 지급]\n'
 '계약자, 피보험자 또는 보험수익자가 보험금 청구에 관한 서류를 변조하여 보험금을 청구한 경우,\n'
 '회사는 그 사실을 안 날부터 1개월 이내에 계약을 해지할 수 있습니다. 다만, 이 경우에도 회사는'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000131',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
