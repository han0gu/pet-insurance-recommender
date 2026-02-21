from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 금융회사(우체국을 포함합니다)를<br>통하여 보험료를 납입한 경우에는 그 금융회사 발행 증빙서류를 영수증으로 '
 "대신합니다.</p><br><h1 id='50' style='font-size:14px'>【납입기일】</h1><br><p id='51' "
 "data-category='paragraph' style='font-size:14px'>계약자가 제2회 이후의 보험료를 납입하기로 한 "
 "날을 말합니다.</p><footer id='52' style='font-size:14px'>- 15 -</footer><h1 "
 "id='53'"),
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
 'indexing': {'chunk_id': 'chunk_000144',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
