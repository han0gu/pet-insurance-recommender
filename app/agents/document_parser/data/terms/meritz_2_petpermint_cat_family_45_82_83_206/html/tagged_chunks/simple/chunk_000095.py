from langchain_core.documents import Document

chunk = Document(
    page_content=("할(또는 반환받을) 금액이 발생할 수 있<br>습니다.</p><h1 id='37' "
 "style='font-size:20px'>【계약자적립액】</h1><br><p id='38' data-category='paragraph' "
 "style='font-size:16px'>장래의 해약환급금 등을 지급하기 위하여 계약자가 납입<br>한 보험료 중 일정액을 기준으로 "
 "보험료 및 해약환급금<br>산출방법서에서 정한 방법에 따라 계산한 금액을 말합니<br>다.</p><br><p id='39' "
 "data-category='paragraph'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000095',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
