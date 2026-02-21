from langchain_core.documents import Document

chunk = Document(
    page_content=('colspan="2">료) 중 다음의 수가코드에 해당하는 검사를 말합니다.</td><td></td></tr><tr><td '
 'rowspan="10">창상봉합술Ⅰ (급여) (안면/경부 외)</td><td>대상이 되는 '
 '항목</td><td>수가코드</td></tr><tr><td>창상봉합술</td><td></td></tr><tr><td>나'),
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
 'indexing': {'chunk_id': 'chunk_001748',
              'chunk_char_len': 184,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
