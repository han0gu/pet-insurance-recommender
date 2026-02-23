from langchain_core.documents import Document

chunk = Document(
    page_content=('- 나 약물을 주입하는 것\n'
 '\uf000 제1항의「수술」은 자택 등에서의 치료가 곤란하여 동물\n'
 '병원에서 행한 것에 한합니다.# 제4조(입원의 정의와 장소)이 계약에 있어서 「입원」이라 함은 수의사가 상해 또는\n'
 '질병의 치료가 필요하다고 인정한 경우로서, 자택 등에서의\n'
 '치료가 곤란하여 동물병원에 입실하여 수의사의 관리 하에\n'
 '치료에 전념하는 것을 말합니다.# 제5조(특별약관의 소멸)이 특별약관에서 정한 보상하는 손해가 더 이상 발생할 수\n'
 '없는 경우에는 이 특별약관은 그 때부터 소멸되며, 이 경우'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000297',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
