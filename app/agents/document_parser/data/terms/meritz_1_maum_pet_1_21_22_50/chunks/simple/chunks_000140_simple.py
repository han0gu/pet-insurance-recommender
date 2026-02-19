from langchain_core.documents import Document

chunk = Document(
    page_content=('제3조(보상하는 손해)\n'
 '① 회사는 피보험자가 대한민국 내에서 이 특별약관의 보험기간 중에 보험증권에 기재된 반려견의 행위에 기인하는 우연한 사고로 인하여 '
 '피해자의 신체의 장해에 대한 법률상 의 배상책임 또는 타인 소유의 반려동물에 손해를 입혀 그에 대한 법률상의 배상책임 을 부담함으로써 '
 '입은 손해(이하「배상책임손해」라 합니다)를 보상합니다. ② 제1항의 피보험자라 함은 보통약관 제3조(피보험자의 범위)를 따릅니다. ③ '
 '1사고당 보상하는 손해의 범위는 아래와 같습니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 22},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000140',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
