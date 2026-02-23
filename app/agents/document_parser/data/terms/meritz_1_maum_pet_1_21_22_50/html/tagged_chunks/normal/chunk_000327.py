from langchain_core.documents import Document

chunk = Document(
    page_content=("회사가 열람을 요구할 경우에는 이에 따라<br>야 합니다.</p><h1 id='81' "
 "style='font-size:14px'>제3조(예치보험료)</h1><br><p id='82' "
 "data-category='paragraph' style='font-size:14px'>예치보험료는 계약체결일 이전 1개월 동안 1일 "
 "평균 보험의 목적의 수에 정해진 보험요율<br>을 적용하여 계산합니다.</p><h1 id='83' "
 "style='font-size:14px'>제4조(보험료의 정산방법)</h1><br><p id='84'"),
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
 'indexing': {'chunk_id': 'chunk_000327',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
