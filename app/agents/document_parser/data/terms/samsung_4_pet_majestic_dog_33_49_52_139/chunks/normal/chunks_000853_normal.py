from langchain_core.documents import Document

chunk = Document(
    page_content=('. ③ 제2항에도 불구하고 보험업법 제97조 제1항 제5호 및 동법 시행령 제43조의2 제1항 에 따른 보장내용 등이 비슷한 '
 '보험계약(이하 「유사계약」이라 합니다)이 계약 청약 일 현재 유지중이거나, 계약 청약일 전 6개월 이내에 계약자 및 피보험자의 요구 또는 '
 '보험료 납입 연체로 해지된 경우 유사계약에서 정한 부담보 기간 종료일 이내에서 계 약의 부담보 기간을 적용하고, 유사계약에서 정한 질병과 '
 '동일하거나 축소된 범위로 계약의 부담보 설정 범위를 정하며, 유사계약이 다수인 경우 피보험자에게 가장 유리 한 계약조건을 적용합니다'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 136},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000853',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
