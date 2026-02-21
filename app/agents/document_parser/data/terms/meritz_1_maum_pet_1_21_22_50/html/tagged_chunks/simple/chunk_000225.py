from langchain_core.documents import Document

chunk = Document(
    page_content=(". 손해배상금 및 그 밖의 비용을 지급하였음을 명하는 서류<br>4. 회사가 요구하는 그 밖의 서류</p><h1 id='44' "
 "style='font-size:14px'>제7조(보험금의 지급절차)</h1><br><p id='45' "
 "data-category='list' style='font-size:14px'>① 회사는 제6조(보험금의 청구)에서 정한 서류를 접수한 "
 '때에는 접수증을 교부하고, 그<br>서류를 접수받은 후 지체없이 지급할 보험금을 결정하고 지급할 보험금이 결정되면 7<br>일 이내에 '
 '이를 지급하여 드립니다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000225',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
