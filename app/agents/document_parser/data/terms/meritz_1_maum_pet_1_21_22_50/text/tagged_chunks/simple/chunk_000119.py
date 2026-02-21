from langchain_core.documents import Document

chunk = Document(
    page_content=('반려견의 행위에 기인하는 우연한 사고로 인하여 피해자의 신체의 장해에 대한 법률상\n'
 '의 배상책임 또는 타인 소유의 반려동물에 손해를 입혀 그에 대한 법률상의 배상책임\n'
 '을 부담함으로써 입은 손해(이하「배상책임손해」라 합니다)를 보상합니다.\n'
 '② 제1항의 피보험자라 함은 보통약관 제3조(피보험자의 범위)를 따릅니다.\n'
 '③ 1사고당 보상하는 손해의 범위는 아래와 같습니다.1. 피보험자가 피해자에게 지급할 책임을 지는 법률상의 손해배상금'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000119',
              'chunk_char_len': 239,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
