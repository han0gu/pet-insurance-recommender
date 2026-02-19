from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 회사는 제2항에 따라 계약내용을 변경할 때 위험이 감소 된 경우에는 보험료를 감액하고, 이후 기간 보장을 위한 재 원인 '
 '계약자적립액 등의 차이로 인하여 발생한 정산금액(이 하 「정산금액」이라 합니다)을 환급하여 드립니다. 한편 위험이 증가된 경우에는 '
 '보험료의 증액 및 정산금액의 추가 납입을 요구할 수 있으며, 계약자는 일시납 또는 잔여 보험 료 납입기간과 5년 중 큰 기간(단, 잔여 '
 '보험기간을 초과할 수 없음) 동안의 분납 중 선택하여 정산금액을 납입하여야 합니다'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 96},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000212',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
