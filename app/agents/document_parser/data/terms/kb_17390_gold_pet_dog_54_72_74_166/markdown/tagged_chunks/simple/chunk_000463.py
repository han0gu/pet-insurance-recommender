from langchain_core.documents import Document

chunk = Document(
    page_content=('기관을 말한다.\n'
 '\uf000 제1항 제4호의 사고증명서는 수의사법 제12조(진단서 등)에서 규정한 내용에 따라# 국내의 동물병원에서 수의사에 의해 발급한 '
 '것이어야 합니다.| 관 련 법 규 | 수의사법 제12조(진단서 등) |\n'
 '| --- | --- |'),
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
 'indexing': {'chunk_id': 'chunk_000463',
              'chunk_char_len': 134,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
