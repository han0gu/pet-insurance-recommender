from langchain_core.documents import Document

chunk = Document(
    page_content=("-</footer><h1 id='28' style='font-size:18px'>보험료 자동납입 특별약관</h1><h1 id='29' "
 "style='font-size:14px'>제1조(보험료의 납입)</h1><br><p id='30' "
 "data-category='paragraph' style='font-size:14px'>계약자는 보험료 분납 특별약관에 의하여 제2회 "
 "이후의 보험료부터 이 특별약관에 따라<br>계약자의 거래은행 지정계좌를 이용하여 보험료를 자동 납입합니다.</p><h1 id='31'"),
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
 'indexing': {'chunk_id': 'chunk_000298',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
