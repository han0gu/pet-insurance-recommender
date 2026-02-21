from langchain_core.documents import Document

chunk = Document(
    page_content=('KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 155- 155 -|  | 유형 제한 정도 배설을 돕기 위해 설치한 의료장치나 외과적 '
 '시술물을 사용함에 있어 타인의 계속적인 도움이 필요한 상태, | 지급률 20% |\n'
 '| --- | --- | --- |'),
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
 'indexing': {'chunk_id': 'chunk_000948',
              'chunk_char_len': 142,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
