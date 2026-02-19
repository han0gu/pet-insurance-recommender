from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 갱신형 펫퍼민트 반려견 배상책임보장 특별약관\n'
 '제1조(보상하는 손해)\n'
 '\uf000 회사는 피보험자가 이 특별약관의 보험기간 중에 보험증 권에 기재된 반려견의 행위에 기인하는 우연한 사고로 인하 여 피해자에게 '
 '신체의 장해에 대한 법률상의 배상책임 또는 타인 소유의 반려동물에 손해를 입혀 그에 대한 법률상의 배상책임을 부담함으로써 입은 '
 '손해(이하「배상책임손해」 라 합니다)를 보상합니다.\n'
 '\uf000 제1항의 피보험자라 함은 아래에 정한 보험증권에 기재\n'
 '된 피보험자 및 그 가족을 말합니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 186},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000630',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
