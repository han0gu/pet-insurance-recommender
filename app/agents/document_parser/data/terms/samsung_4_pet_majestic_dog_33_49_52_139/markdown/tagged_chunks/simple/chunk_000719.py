from langchain_core.documents import Document

chunk = Document(
    page_content=('- 2. 감액완납보험으로의 변경\n'
 '<용어풀이>[감액완납보험]\n'
 '차회 이후의 보험료 납입을 중단하는 대신 가입금액을 감액하는 보험제6조 (준용규정)이 특별약관에 정하지 않은 사항은 보험계약을 '
 '따릅니다.- 134 -5-4. 특정 신체부위·질병 보장제한부 인수 특별약관# 제 1조 (계약의 체결 및 효력)① 이 특별약관은 '
 '보험계약(특별약관이 부가된 경우에는 특별약관을 포함합니다. 이하\n'
 '「보험계약」이라 합니다)을 체결 또는 변경할 때 다음 각 호의 경우 보험계약자(이하'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000719',
              'chunk_char_len': 257,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
