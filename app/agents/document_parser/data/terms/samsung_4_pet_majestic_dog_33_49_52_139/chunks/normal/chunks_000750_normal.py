from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[유익하였던 비용]\n'
 '물건의 개량∙이용을 위하여 지출되는 비용으로, 물건의 가치를 증가시키는 데 도움이 되는 비용을 말합니다.\n'
 '[공탁보증보험료]\n'
 '가압류, 가집행, 가처분 등 각종 민사사건을 신청할 때, 잘못된 신청으로 인해 발생하는 피신청인 의 손해를 법적으로 보상해 주기 위해 '
 '법원에 납부하는 공탁금을 대신하는 보험상품을 공탁보증보 험이라 하며, 이 보험에 가입하기 위해 필요한 보험료를 공탁보증보험료라 '
 '말합니다.\n'
 '제4조 (보상하지 않는 손해)'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 120},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000750',
              'chunk_char_len': 255,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
