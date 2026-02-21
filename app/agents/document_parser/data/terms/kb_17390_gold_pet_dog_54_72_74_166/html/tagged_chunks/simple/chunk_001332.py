from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>\uf000 계약자와 동일한 피보험자에 대해서만 이 선지급서비스 "
 '특별약관(이하"특별약관"<br>이라 합니다)을 부가할 수 있습니다.<br>\uf000 이 특별약관의 보험기간은 보통약관의 보험기간이 '
 '끝나는 날의 12개월 이전까지로<br>합니다.<br>\uf000 보통약관에 사망보험금을 지급하는 특별약관(이하 "사망보장특별약관"이라 '
 "합니<br>다)이 부가되어 있는 경우에도 이 특별약관을 적용합니다.</p><br><h1 id='200' "
 "style='font-size:14px'>제2조(지급사유)</h1><br><h1"),
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
 'indexing': {'chunk_id': 'chunk_001332',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
