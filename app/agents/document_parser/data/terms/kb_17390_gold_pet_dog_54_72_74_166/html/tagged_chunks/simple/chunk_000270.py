from langchain_core.documents import Document

chunk = Document(
    page_content=("id='83' data-category='paragraph' style='font-size:14px'>위법계약<br>위법계약이라 함은 "
 '｢금융소비자보호에 관한 법률｣ 제47조에서 정한 적합성원<br>칙, 적정성원칙, 설명의무, 불공정영업행위 금지 또는 부당권유행위 금지를 '
 "위</p><br><p id='84' data-category='list'></p><br><h1 id='85' "
 "style='font-size:14px'>반한 계약을</h1><br><h1 id='86'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000270',
              'chunk_char_len': 261,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
