from langchain_core.documents import Document

chunk = Document(
    page_content=('회사는 아래의 사유를 원인으로 하여 생긴 손해는 보상하지 않습니다.\n'
 '1. 특별약관 일반사항 제7조(보험금을 지급하지 않는 사유) 제1항 제1호 및 제2호 2. 피보험자의 배우자 및 직계존비속에 의한 손해 '
 '3. 피보험자가 범죄행위를 하던 중 또는 폭력행위 등 처벌에 관한 법률 제4조의 범죄 단체를 구성 또는 이에 가담함으로써 발생된 손해 '
 '4. 전쟁, 폭동, 소요, 노동쟁의 또는 이와 유사한 사변 중에 생긴 손해 5. 피보험자와 고용관계에 있는 고용주 내지 고용상의 '
 '관리책임이 있는 자에 의해 발 생한 손해\n'
 '- 86 -'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 87},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000465',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
