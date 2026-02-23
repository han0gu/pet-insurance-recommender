from langchain_core.documents import Document

chunk = Document(
    page_content=('기간 중에 사망한 경우 보험증권에 기재된 보험가입금액을 보험수익자에게 보상하여\n'
 '드립니다.\n'
 '② 제1항의 사망은 동물병원에서 적법하게 시행된 안락사를 포함합니다. 단, 이 경우 동\n'
 '물병원에서 발급한 소견서를 제출하여야 합니다.\n'
 '③ 제1항의 손해에 대한 보장개시일(책임개시일)은 이 특별약관의 보험계약일(이하 「보\n'
 '험계약일」이라 합니다)부터 그 날을 포함하여 30일이 지난 날의 다음날로 합니다. 이\n'
 '경우 보험계약일은 이 특별약관의 제1회 보험료를 받은 날로 합니다.-'),
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
 'indexing': {'chunk_id': 'chunk_000619',
              'chunk_char_len': 259,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
