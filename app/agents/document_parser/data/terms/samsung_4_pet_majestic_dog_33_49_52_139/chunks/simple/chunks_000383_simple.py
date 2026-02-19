from langchain_core.documents import Document

chunk = Document(
    page_content=('. 단, 당일입원하여 수 술하는 경우는 상해 입원 수술비를 보상하지 않고 제2항을 따릅니다. ② 회사는 피보험자가 보험기간 중에 상해의 '
 '직접적인 결과로써 병원 또는 의원에 외래 로 방문하거나 당일입원하여 수술(이하 「상해 통원 수술」 이라 합니다)을 받은 경우에 는 수술 '
 '1회당 보험증권에 기재된 이 특별약관 해당 세부보장의 보험가입금액을 상해 통원 수술비(외래 및 당일입원)(이하 「상해 통원 수술비」 라 '
 '합니다)로 보험수익자에 게 지급합니다 ③ 제1항 및 제2항의 「당일입원」 이라 함은 입원일자와 퇴원일자가 동일한 입원을 말합 니다'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 75},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000383',
              'chunk_char_len': 300,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
