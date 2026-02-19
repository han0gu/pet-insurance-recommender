from langchain_core.documents import Document

chunk = Document(
    page_content=('제 3조 (보상하는 손해)\n'
 '① 회사는 대한민국 내에서 보험증권에 기재된 이 특별약관의 보험기간(이하「보험기간 」이라 합니다) 중에 보험증권에 기재된 피보험자의 '
 '반려견의 행위에 기인하는 우연 한 사고(이하「사고」라 합니다)로 인하여 타인의 신체에 피해를 입히거나 타인 소유 의 반려동물에 손해를 '
 '입혀 법률상의 배상책임을 부담함으로써 입은 손해(이하「배상 책임손해」라 합니다)를 이 특별약관에 따라 보상합니다. ② 제1항에서 회사가 '
 '1사고당 보상하는 손해의 범위는 아래와 같습니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 120},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000746',
              'chunk_char_len': 267,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
