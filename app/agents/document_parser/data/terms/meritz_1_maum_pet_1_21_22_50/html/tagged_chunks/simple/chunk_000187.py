from langchain_core.documents import Document

chunk = Document(
    page_content=("data-category='list' style='font-size:14px'>② 회사는 약관의 뜻이 명백하지 않은 경우에는 계약자에게 "
 '유리하게 해석합니다.<br>③ 회사는 보험금을 지급하지 않는 사유 등 계약자나 피보험자에게 불리하거나 부담을 주<br>는 내용은 확대하여 '
 "해석하지 않습니다.</p><h1 id='112' style='font-size:14px'>제38조(설명서 교부 및 보험안내자료 등의 "
 "효력)</h1><br><p id='113' data-category='list' style='font-size:14px'>①"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000187',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
