from langchain_core.documents import Document

chunk = Document(
    page_content=('| 창상봉합술Ⅱ (급여) (안면/경부) 160 | 주: 길이 10cm이상 창상봉합술을 시행할경우 소 정점수에 63.81점을 가산하며, '
 '창상봉합 길 이가 5cm 증가될때마다 63.81점을 추가 가산 한다. (나) 근육에 달하는것 | SA030 |\n'
 '| 창상봉합술Ⅱ (급여) (안면/경부) 160 |  |  |\n'
 '| 창상봉합술Ⅱ (급여) (안면/경부) 160 | 1) 길이 1.5cm 미만 | SA031 |\n'
 '| 창상봉합술Ⅱ (급여) (안면/경부) 160 | 2) 길이 1.5cm 이상 ~ 3.0cm 미만 | SA032 |'),
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
 'indexing': {'chunk_id': 'chunk_000990',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
