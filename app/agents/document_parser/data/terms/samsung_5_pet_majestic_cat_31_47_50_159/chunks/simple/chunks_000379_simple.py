from langchain_core.documents import Document

chunk = Document(
    page_content=('① 회사는 피보험자가 보험증권에 기재된 이 특별약관의 보험기간(이하 「보험기간」 이라 합니다) 중에 상해의 직접적인 결과로써 병원 또는 '
 '의원에 2일이상 입원하여 수술(이 하 「상해 입원 수술」 이라 합니다)을 받은 경우에는 수술 1회당 보험증권에 기재된 이 특별약관 해당 '
 '세부보장의 보험가입금액을 상해 입원 수술비(당일입원 제외)(이하 「상해 입원 수술비」 라 합니다)로 보험수익자에게 지급합니다. 단, '
 '당일입원하여 수 술하는 경우는 상해 입원 수술비를 보상하지 않고 제2항을 따릅니다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 73},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000379',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
