from langchain_core.documents import Document

chunk = Document(
    page_content=('4-5. [갱신형] 반려견 배상책임보장 특별약관\n'
 '제 1조 (목적)\n'
 '이 특별약관은 보험계약자(이하「계약자」라 합니다)와 보험회사(이하「회사」라 합니다) 사이에 피보험자가 법률상의 배상책임을 부담함으로써 '
 '입은 손해에 대한 위험을 보장하 기 위하여 체결됩니다.\n'
 '제 2조 (용어의 정의)\n'
 '이 특별약관에서 사용되는 용어의 정의는 이 특별약관의 다른 조항에서 달리 정의되지 않는 한 다음과 같습니다.\n'
 '① 계약관계 관련 용어'),
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
 'indexing': {'chunk_id': 'chunk_000740',
              'chunk_char_len': 230,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
