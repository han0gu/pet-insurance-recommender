from langchain_core.documents import Document

chunk = Document(
    page_content=('. 안면 또는 경부 이외 (1)</td><td>단순봉합</td></tr><tr><td>(가) 표재성인 '
 '것</td><td></td></tr><tr><td>1) 길이 2.5cm 미만</td><td>SB021</td></tr><tr><td>2) '
 '길이 2.5cm 이상 ~ 5.0cm 미만</td><td>SB022</td></tr><tr><td>(2) 변연절제를 '
 '포함</td><td></td></tr><tr><td>(가) 표재성인 것</td><td></td></tr><tr><td>1) 길이 '
 '2.5cm'),
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
 'indexing': {'chunk_id': 'chunk_001749',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
