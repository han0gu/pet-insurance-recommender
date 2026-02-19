from langchain_core.documents import Document

chunk = Document(
    page_content=('③ 회사는 제2항에 따라 특별약관 내용을 변경할 때 위험이 감소된 경우에는 보험료를 감액하고, 이후 기간 보장을 위한 재원인 계약자적립액 '
 '등의 차이로 인하여 발생한 정 산금액(이하 「정산금액」이라 합니다)을 환급하여 드립니다. 한편 위험이 증가된 경 우에는 보험료의 증액 및 '
 '정산금액의 추가납입을 요구할 수 있으며, 계약자는 일시납 또는 잔여 보험료 납입기간과 5년 중 큰 기간(단, 잔여 보험기간을 초과할 수 '
 '없음) 동안의 분납 중 선택하여 정산금액을 납입하여야 합니다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 54},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000220',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
