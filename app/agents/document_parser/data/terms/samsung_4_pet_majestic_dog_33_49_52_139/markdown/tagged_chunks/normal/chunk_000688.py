from langchain_core.documents import Document

chunk = Document(
    page_content=('- 3. 위생관리, 미모를 위한 성형수술\n'
 '- 4. 정상분만, 치과질환\n'
 '# 제7조 (특별약관의 소멸)피보험자 또는 보험증권에 기재된 반려견이 보험기간 중에 사망하였을 경우에는 "보험료\n'
 '및 해약환급금 산출방법서"에서 정하는 바에 따라 회사가 적립한 사망당시 이 특별약관의\n'
 '계약자적립액 및 미경과보험료를 계약자에게 지급하고, 이 특별약관은 더 이상 효력이 없\n'
 '습니다.# 제8조 (특별약관의 자동갱신)이 특별약관은 제도성 특별약관 5-1. [갱신형] 특별약관의 자동갱신 특별약관에 따라 갱'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000688',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
