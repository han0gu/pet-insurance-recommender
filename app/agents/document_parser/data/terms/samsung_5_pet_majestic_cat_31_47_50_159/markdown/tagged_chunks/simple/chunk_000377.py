from langchain_core.documents import Document

chunk = Document(
    page_content=('- 합니다) 중에 상해를 입고 그 직접적인 결과로써 생활기능 또는 업무능력에 지장을 가\n'
 '- 져와 병원 또는 의원(한방병원 또는 한의원을 포함합니다)에 1일이상 계속 입원하여\n'
 '- 치료를 받은 경우에는 입원 1일당 보험증권에 기재된 이 특별약관의 보험가입금액을\n'
 '- 상해 입원일당(1일이상)으로 보험수익자에게 지급합니다. 다만, 상해 입원일당(1일이\n'
 '- 상)의 지급일수는 1회 입원당 180일을 한도로 합니다.\n'
 '- ② 제1항의 경우 피보험자가 동일한 상해의 치료를 직접적인 목적으로 2회 이상 입원한'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000377',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
