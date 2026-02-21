from langchain_core.documents import Document

chunk = Document(
    page_content=('잃은 합의로 보험수익<br>자에게 손해를 가한 경우에도 회사는 제2항에 따라 손해를 배상할 책임을 집니다.</p><br><h1 '
 "id='2' style='font-size:14px'>【현저하게 공정을 잃은 합의】</h1><br><p id='3' "
 "data-category='paragraph' style='font-size:14px'>사회통념상 일반 보통인이라면 그 같은 일을 하지 "
 "않을 정도로 현저하게 공정성을<br>잃은 것을 말합니다.</p><h1 id='4'"),
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
 'indexing': {'chunk_id': 'chunk_000192',
              'chunk_char_len': 257,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
