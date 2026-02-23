from langchain_core.documents import Document

chunk = Document(
    page_content=("정한 바에 따라 일단위로 계산하여 받거나 돌려 드립니다.</p><h1 id='69' "
 "style='font-size:14px'>제6조(보험증권의 발급)</h1><br><p id='70' data-category='list' "
 "style='font-size:14px'>① 회사는 보험계약자에게 보험증권을 드려야 하고, 그 약관의 주요한 내용을 "
 "알려드립니다.<br>② 보험계약자의 요청이 있을 경우, 개별 피보험자에게는 가입증명서를 발급하여 드립니다.</p><h1 id='71' "
 "style='font-size:14px'>제7조(적용상의"),
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
 'indexing': {'chunk_id': 'chunk_000322',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
