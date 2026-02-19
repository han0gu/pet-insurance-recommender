from langchain_core.documents import Document

chunk = Document(
    page_content=('정한 손해에 대한 보장개시일(책임개시일)의 전일 이전에 사망한 경우에는 "보험료 및 해약환급금 산출방법서"에서 정하는 바에 따라 회사가 '
 '적립한 부활(효력회복)일 당시의 이 특별약관의 계약자적립액 및 미경과보험료를 지급하고, 부활(효력회복) 일 이후부터 납입한 이 특별약관의 '
 '보험료를 돌려 드립니다.\n'
 '제 5조 (특별약관의 소멸)'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 119},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000735',
              'chunk_char_len': 182,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
