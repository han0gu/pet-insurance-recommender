from langchain_core.documents import Document

chunk = Document(
    page_content=("발생을 막을 수 있었음에도 그 주의<br>조차 태만히 한 높은 강도의 주의의무 위반(이하 같습니다.)</p><br><p id='42' "
 "data-category='list' style='font-size:14px'>2"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000032',
              'chunk_char_len': 120,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
