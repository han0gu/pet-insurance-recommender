from langchain_core.documents import Document

chunk = Document(
    page_content=('원)」 의 총 2개의 세부보장으로 구성되어 있습니다.# 제2조 (보험금의 지급사유)- ① 회사는 피보험자가 보험증권에 기재된 이 '
 '특별약관의 보험기간(이하 「보험기간」 이라\n'
 '- 합니다) 중에 상해의 직접적인 결과로써 병원 또는 의원에 2일이상 입원하여 수술(이\n'
 '- 하 「상해 입원 수술」 이라 합니다)을 받은 경우에는 수술 1회당 보험증권에 기재된\n'
 '- 이 특별약관 해당 세부보장의 보험가입금액을 상해 입원 수술비(당일입원 제외)(이하\n'
 '- 「상해 입원 수술비」 라 합니다)로 보험수익자에게 지급합니다. 단, 당일입원하여 수'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000322',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
