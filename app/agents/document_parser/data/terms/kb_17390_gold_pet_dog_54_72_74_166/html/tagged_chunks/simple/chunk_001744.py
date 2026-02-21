from langchain_core.documents import Document

chunk = Document(
    page_content=('. (나) 근육에 달하는것</td><td>SA030</td></tr><tr><td></td><td></td></tr><tr><td>1) '
 '길이 1.5cm 미만</td><td>SA031</td></tr><tr><td>2) 길이 1.5cm 이상 ~ 3.0cm '
 '미만</td><td>SA032</td></tr><tr><td>3) 길이 3.0cm 이상 ~ 5.0cm '
 '미만</td><td>SA037</td></tr><tr><td>4) 길이 5.0cm 이상 ~ 7.5cm '
 '미만</td><td>SA038</td></tr><tr><td>5) 길이'),
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
 'indexing': {'chunk_id': 'chunk_001744',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
