from langchain_core.documents import Document

chunk = Document(
    page_content=("제12조(진단서 등)】</p><br><p id='62' data-category='list' style='font-size:14px'>① "
 '수의사는 자기가 직접 진료하거나 검안하지 아니하고는 진단서, 검안서, 증명서<br>또는 처방전(「전자서명법」에 따른 전자서명이 기재된 '
 '전자문서 형태로 작성한<br>처방전을 포함한다'),
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
 'indexing': {'chunk_id': 'chunk_000054',
              'chunk_char_len': 177,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
