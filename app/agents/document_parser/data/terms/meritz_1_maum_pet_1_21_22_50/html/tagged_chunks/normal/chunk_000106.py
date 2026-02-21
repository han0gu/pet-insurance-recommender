from langchain_core.documents import Document

chunk = Document(
    page_content=("수 있습니다.</p><p id='6' data-category='paragraph' style='font-size:14px'>제 4 관 "
 "보험계약의 성립과 유지</p><p id='7' data-category='paragraph' "
 "style='font-size:14px'>제19조(보험계약의 성립)</p><br><p id='8' data-category='list' "
 "style='font-size:14px'>① 계약은 계약자의 청약과 회사의 승낙으로 이루어집니다.<br>② 회사는 피보험자가 계약에 "
 '적합하지 않은 경우에는 승낙을'),
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
 'indexing': {'chunk_id': 'chunk_000106',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
