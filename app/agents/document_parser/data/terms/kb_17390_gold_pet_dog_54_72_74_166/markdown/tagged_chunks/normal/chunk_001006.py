from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- | --- |\n'
 '| 바이러스성 폐렴 | 파라인플루엔자바이러스폐렴 | J12.2 |\n'
 '| 바이러스성 폐렴 | 사람메타뉴모바이러스폐렴 | J12.3 |\n'
 '| 바이러스성 폐렴 |  |  |\n'
 'KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 161- 161 -|  |  |  |\n'
 '| --- | --- | --- |\n'
 '| 특정세균성 | 대상이 되는 항목 | 분류번호 |\n'
 '| 특정세균성 | 폐렴연쇄알균에 의한 폐렴 폐렴 | J13 |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001006',
              'chunk_char_len': 245,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
