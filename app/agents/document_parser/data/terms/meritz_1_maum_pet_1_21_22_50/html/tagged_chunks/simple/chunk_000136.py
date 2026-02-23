from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>제5관 보험료의 납입</p><h1 id='41' "
 "style='font-size:14px'>제25조(제1회 보험료 및 회사의 보장개시)</h1><br><p id='42' "
 "data-category='list' style='font-size:14px'>① 회사는 계약의 청약을 승낙하고 제1회 보험료를 받은 "
 '때부터 이 약관이 정한 바에 따<br>라 보장을 합니다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000136',
              'chunk_char_len': 222,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
