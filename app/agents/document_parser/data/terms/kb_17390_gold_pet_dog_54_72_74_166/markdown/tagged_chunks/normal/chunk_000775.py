from langchain_core.documents import Document

chunk = Document(
    page_content=('- 18세 갱신시점에서는 20세 갱신종료시까지의 잔여보험기간이 3년보다 작\n'
 '아 3년만기로 갱신하지 않고 2년만기로 갱신합니다.\uf000 이 보장특약이 정상적으로 유지되고 다음 각 호의 조건을 충족하는 경우에는 '
 '보장\n'
 '특약의 만기되는 날의 전일까지 계약자의 별도의 의사표시가 없을 때에는 종전의\n'
 '보장특약(이하 "갱신전 보장특약"이라 합니다)과 동일한 내용으로 보장특약의 만\n'
 '기일의 다음날(이하 "갱신일"이라 합니다)에 갱신되는 것으로 합니다.# 1. 갱신된 특별약관(이하 "갱신보장특약"이라 합니다)의 만기일이 '
 '회사가 정한- 기간 내일 것'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000775',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
