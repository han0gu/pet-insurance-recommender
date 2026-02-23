from langchain_core.documents import Document

chunk = Document(
    page_content=('| 창상봉합술Ⅱ (급여) (안면/경부 | 1) 길이 2.5cm 미만 | SC031 |\n'
 '| 창상봉합술Ⅱ (급여) (안면/경부 | 2) 길이 2.5cm 이상 ~ 5.0cm 미만 | SC032 |\n'
 '| 창상봉합술Ⅱ (급여) (안면/경부 | 3) 길이 5.0cm 이상 ~ 10.0cm 미만 | SC039 |\n'
 '| 창상봉합술Ⅱ (급여) (안면/경부 |  |  |'),
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
 'indexing': {'chunk_id': 'chunk_001000',
              'chunk_char_len': 195,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
