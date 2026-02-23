from langchain_core.documents import Document

chunk = Document(
    page_content=('| 창상봉합술Ⅰ (급여) (안면/경부 외) | (2) 변연절제를 포함 |  |\n'
 '| 창상봉합술Ⅰ (급여) (안면/경부 외) | (가) 표재성인 것 |  |\n'
 '| 창상봉합술Ⅰ (급여) (안면/경부 외) | 1) 길이 2.5cm 미만 | SC021 |\n'
 '| 창상봉합술Ⅰ (급여) (안면/경부 외) | 2) 길이 2.5cm 이상 ~ 5.0cm 미만 | SC022 |\n'
 '160 -|  |  |  |\n'
 '| --- | --- | --- |\n'
 '| 창상봉합술Ⅱ (급여) (안면/경부 | 대상이 되는 항목 | 수가코드 |'),
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
 'indexing': {'chunk_id': 'chunk_000995',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
