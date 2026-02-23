from langchain_core.documents import Document

chunk = Document(
    page_content=('. 피보험자가 제12조(손해배상청구에 대한 회사의 해결)의 제2항 및 제3항의 회사<br>의 요구에 따르기 위하여 지출한 '
 "비용</p><br><h1 id='29' style='font-size:14px'>【공탁보증보험료】</h1><br><p id='30' "
 "data-category='paragraph' style='font-size:14px'>가압류, 가집행, 가처분 신청 등 각종 민사사건을 "
 '신청할 때, 잘못된 신청으로<br>인해 발생하는 피신청인의 손해를 법적으로 보상해 주기 위해서 법원에 납부하<br>는 공탁금을 대신하는 '
 '보험상품의'),
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
 'indexing': {'chunk_id': 'chunk_000211',
              'chunk_char_len': 300,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
