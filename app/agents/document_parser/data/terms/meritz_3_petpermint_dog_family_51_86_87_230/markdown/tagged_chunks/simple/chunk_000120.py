from langchain_core.documents import Document

chunk = Document(
    page_content=('- 그 서류 또는 증거를 위조 또는 변조한 경우. 다만,\n'
 '- 이미 보험금 지급사유가 발생한 경우에는 이에 대한\n'
 '80# 보험금은 지급합니다.# 【 예시 】입원특약에 가입한 피보험자가 20일간 입원하였음에도 불\n'
 '구하고 입원확인서를 변조하여 입원일수 30일에 해당하는\n'
 '보험금을 청구한 경우, 회사는 그 사실을 안 날로부터 1\n'
 '개월 이내에 계약을 해지할 수 있습니다. 다만, 이 경우\n'
 '에도 회사는 입원일수 20일에 해당하는 보험금을 지급합\n'
 '니다.\uf000 회사가 제1항에 따라 계약을 해지한 경우 회사는 그 취'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000120',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
