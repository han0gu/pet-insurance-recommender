from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 계약자가 제2항에 따라 보험수익자를 변경하고자 할 경 우 계약자와 피보험자가 동일하지 않을 때에는 보험금 지급 사유가 '
 '발생하기 전에 피보험자가 서면(「전자서명법」 제2 조 제2호에 따른 전자서명이 있는 경우로서 상법 시행령 제 44조의2에 정하는 바에 '
 '따라 본인 확인 및 위조ㆍ변조 방지 에 대한 신뢰성을 갖춘 전자문서를 포함)으로 동의하여야 합니다. \uf000 회사는 제1항에 따라 '
 '계약자를 변경한 경우, 변경된 계 약자에게 보험증권 및 약관을 교부하고 변경된 계약자가 요 청하는 경우 약관의 중요한 내용을 설명하여 '
 '드립니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 73},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000110',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
