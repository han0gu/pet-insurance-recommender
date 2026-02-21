from langchain_core.documents import Document

chunk = Document(
    page_content=(". 제7항에 따라 보험금 지급사유에 대해 제3자의 의견에 따르기로 한 경우</p><br><p id='68' "
 "data-category='paragraph' style='font-size:14px'>③ 제2항에 의하여 추가적인 조사가 이루어지는 "
 '경우, 회사는 보험수익자의 청구에 따라<br>회사가 추정하는 보험금의 50% 상당액을 가지급보험금으로 지급합니다.</p><br><p '
 "id='69' data-category='paragraph' style='font-size:14px'>【가지급보험금】</p><br><p "
 "id='70'"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000062',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
