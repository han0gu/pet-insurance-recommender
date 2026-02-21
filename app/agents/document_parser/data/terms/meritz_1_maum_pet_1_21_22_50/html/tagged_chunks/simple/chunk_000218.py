from langchain_core.documents import Document

chunk = Document(
    page_content=(". 가입 반려견이 질병을 전염시켜 발생한 배상책임</p><footer id='36' style='font-size:14px'>- 23 "
 "-</footer><h1 id='37' style='font-size:14px'>제5조(손해의 통지 및 조사)</h1><br><p "
 "id='38' data-category='paragraph' style='font-size:14px'>① 계약자 또는 피보험자는 아래와 "
 "같은 사실이 있는 경우에는 지체없이 그 내용을 회사에<br>알려야 합니다.</p><br><p id='39'"),
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
 'indexing': {'chunk_id': 'chunk_000218',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
