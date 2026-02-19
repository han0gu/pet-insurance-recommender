from langchain_core.documents import Document

chunk = Document(
    page_content=('① 회사는 피보험자가 보험증권에 기재된 이 특별약관의 보험기간(이하「보험기간」이라 합니다) 중에 상해를 입고 그 직접적인 결과로써 '
 '생활기능 또는 업무능력에 지장을 가 져와 병원 또는 의원(한방병원 또는 한의원을 포함합니다)에 1일이상 계속 입원하여 치료를 받은 '
 '경우에는 입원 1일당 보험증권에 기재된 이 특별약관의 보험가입금액을 상해 입원일당(1일이상)으로 보험수익자에게 지급합니다. 다만, 상해 '
 '입원일당(1일이 상)의 지급일수는 1회 입원당 180일을 한도로 합니다'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 85},
 'term_type': 'special',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000450',
              'chunk_char_len': 261,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
