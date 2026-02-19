from langchain_core.documents import Document

chunk = Document(
    page_content=('제21조(중대사유로 인한 해지)\n'
 '\uf000 회사는 아래와 같은 사실이 있을 경우에는 안 날부터 1 개월 이내에 계약을 해지할 수 있습니다.\n'
 '① 계약자, 피보험자 또는 보험수익자가 보험금을 지급받 을 목적으로 고의로 보험금 지급사유를 발생시킨 경 우 ② 계약자, 피보험자 또는 '
 '보험수익자가 보험금 청구에 관한 서류에 고의로 사실과 다른 것을 기재하였거나 그 서류 또는 증거를 위조 또는 변조한 경우. 다만, 이미 '
 '보험금 지급사유가 발생한 경우에는 이에 대한 보험금은 지급합니다.\n'
 '【 예시 】'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 107},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000275',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
