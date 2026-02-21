from langchain_core.documents import Document

chunk = Document(
    page_content=('| 창상봉합술Ⅱ (급여) (안면/경부) 160 | 창상봉합술 |  |\n'
 '| 창상봉합술Ⅱ (급여) (안면/경부) 160 | 가. 안면 또는 경부 |  |\n'
 '| 창상봉합술Ⅱ (급여) (안면/경부) 160 | (1) 단순봉합 표재성인 것 |  |\n'
 '| 창상봉합술Ⅱ (급여) (안면/경부) 160 | (가) 3) 길이 3.0cm 이상 ~ 5.0cm 미만 | S0027 |\n'
 '| 창상봉합술Ⅱ (급여) (안면/경부) 160 | 4) 길이 5.0cm 이상 ~ 7.5cm 미만 | S0028 |'),
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
 'indexing': {'chunk_id': 'chunk_000985',
              'chunk_char_len': 264,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
