from langchain_core.documents import Document

chunk = Document(
    page_content=('| 창상봉합술Ⅰ (급여) (안면/경부) | 1) 길이 1.5cm 미만 SA021 | 특별 |\n'
 '| 창상봉합술Ⅰ (급여) (안면/경부) | 2) 길이 1.5cm 이상 ~ 3.0cm 미만 SA022 | 약 |\n'
 '관KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 159별표법ㆍ규정|  |  |  |\n'
 '| --- | --- | --- |\n'
 '| 창상봉합술Ⅱ (급여) (안면/경부) 160 | 대상이 되는 항목 | 수가코드 |\n'
 '| 창상봉합술Ⅱ (급여) (안면/경부) 160 | 창상봉합술 |  |'),
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
 'indexing': {'chunk_id': 'chunk_000984',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
