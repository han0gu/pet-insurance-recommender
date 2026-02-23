from langchain_core.documents import Document

chunk = Document(
    page_content=('로써 회사의 보험계약 안내자료 제공의무를 다한 것으로 보며, 전자적 주소를 사- \n'
 '실과 다르게 알리거나 알리지 않아 발생하는 불이익은 계약자가 부담합니다.- \n'
 '제5조(준용규정)이 특별약관에서 정하지 않은 사항은 보통약관 및 해당 특별약관을 따릅니다.- \n'
 '6. 장애인전용보험전환- 제1조(적용범위)\n'
 '\uf000 이 특별약관은 회사가 정한 방법에 따라 계약자가 청약(請約)하고 회사가 승낙(承\n'
 '諾)함으로써 다음 각 호의 조건을 모두 만족하는 보험계약(이하 "전환대상계약"이'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000794',
              'chunk_char_len': 257,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
