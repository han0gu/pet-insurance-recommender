from langchain_core.documents import Document

chunk = Document(
    page_content=('# 이 초회보험료자동납입 추가특별약관에 정하지 않은 사항은 보통약관 및 보험료자동# 납입 특별약관을 따릅니다.# 4. 보장특약 '
 '자동갱신(추가납입형)제1조(특약의 적용)\n'
 '이 "보장특약 자동갱신(추가납입형) 특별약관"(이하 "특약"이라 합니다)은 손해의\n'
 '보상을 내용으로 한 이 계약의 특별약관(이하 "보장특약"이라 합니다)의 자동갱신에# 대하여 회사와 계약자간에 사전에 합의가 있을 경우에 '
 '적용합니다.제2조(보장특약의 자동갱신)\n'
 '\uf000 이 보장특약의 보험기간은 갱신전 보장특약의 보험기간으로 합니다. 다만, 이 특'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000773',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
