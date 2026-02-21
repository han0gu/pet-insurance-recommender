from langchain_core.documents import Document

chunk = Document(
    page_content=('하고, 이후 기간 보장을 위한 재원인 해약환급금 등의 차이로 인하여 발생한 정산금액(이\n'
 '하 “정산금액”이라 합니다)을 환급하여 드립니다. 한편 위험이 증가된 경우에는 납입보험\n'
 '료의 증액 및 정산금액의 추가납입을 요구할 수 있으며, 계약자는 일시납 또는 잔여 보험\n'
 '료 납입기간과 5년 중 큰 기간(단, 잔여 보험기간을 초과할 수 없음) 동안의 분납 중 선택\n'
 '하여 정산금액을 납입하여야 합니다. 다만, 보험료 갱신형 계약 등 일부 보험계약의 경우\n'
 '분납이 제한될 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000051',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
