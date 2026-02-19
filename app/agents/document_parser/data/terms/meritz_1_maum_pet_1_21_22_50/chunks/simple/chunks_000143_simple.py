from langchain_core.documents import Document

chunk = Document(
    page_content=('【공탁보증보험료】\n'
 '가압류, 가집행, 가처분 신청 등 각종 민사사건을 신청할 때, 잘못된 신청으로 인해 발생하는 피신청인의 손해를 법적으로 보상해 주기 '
 '위해서 법원에 납부하 는 공탁금을 대신하는 보험상품의 보험료를 말합니다.\n'
 '제4조(보상하지 않는 손해)\n'
 '회사는 아래의 사유를 원인으로 하여 생긴 배상책임을 부담함으로써 입은 손해는 보상하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 23},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000143',
              'chunk_char_len': 194,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
