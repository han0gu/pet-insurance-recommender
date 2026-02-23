from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- |\n'
 '| 예 시 ․ 개 : 파보바이러스감염증, 디스템퍼바이러스감염증, 파라인플루엔자감염 증, 전염성 간염, 아데노바이러스2형감염증, '
 '코로나바이러스감염 증, 렙토스피라감염증, 필라리아감염증, 광견병, 인플루엔자 감염, | 예 시 ․ 개 : 파보바이러스감염증, '
 '디스템퍼바이러스감염증, 파라인플루엔자감염 증, 전염성 간염, 아데노바이러스2형감염증, 코로나바이러스감염 증, 렙토스피라감염증, '
 '필라리아감염증, 광견병, 인플루엔자 감염, |\n'
 '- 켄넬코프\n'
 '- 3. 상병명을 알 수 없는 상해 또는 질병에 대한 치료'),
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
 'indexing': {'chunk_id': 'chunk_000561',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
