from langchain_core.documents import Document

chunk = Document(
    page_content=('9세, 12세, 15세, 18세<br>- 18세 갱신시점에서는 20세 갱신종료시까지의 잔여보험기간이 3년보다 작<br>아 3년만기로 '
 "갱신하지 않고 2년만기로 갱신합니다.</p><br><p id='17' data-category='paragraph' "
 "style='font-size:14px'>\uf000 이 보장특약이 정상적으로 유지되고 다음 각 호의 조건을 충족하는 경우에는 "
 '보장<br>특약의 만기되는 날의 전일까지 계약자의 별도의 의사표시가 없을 때에는 종전의<br>보장특약(이하 "갱신전 보장특약"이라 '
 '합니다)과 동일한 내용으로 보장특약의'),
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
 'indexing': {'chunk_id': 'chunk_001369',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
