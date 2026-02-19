from langchain_core.documents import Document

chunk = Document(
    page_content=('② 보험계약자 또는 피보험자나 보험수익자가 제1항의 통 지의무를 해태함으로 인하여 손해가 증가된 때에는 보 험자는 그 증가된 손해를 '
 '보상할 책임이 없다.\n'
 '【공탁보증보험료】\n'
 '가압류, 가집행, 가처분 신청 등 각종 민사사건을 신청할 때, 잘못된 신청으로 인해 발생하는 피신청인의 손해를 법적으로 보상해 주기 '
 '위해서 법원에 납부하는 공탁금을 대신하는 보험상품의 보험료를 말합니다.\n'
 '제4조(보험금의 청구)\n'
 '피보험자가 보험금을 청구할 때에는 다음의 서류를 회사에 제출하여야 합니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 177},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000590',
              'chunk_char_len': 264,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
