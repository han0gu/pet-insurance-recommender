from langchain_core.documents import Document

chunk = Document(
    page_content=("미경과보험료를 계약자에게 지급합니다.</p><br><h1 id='36' "
 "style='font-size:18px'>【계약자적립액】</h1><br><p id='37' data-category='paragraph' "
 "style='font-size:16px'>장래의 해약환급금 등을 지급하기 위하여 계약자가 납입<br>한 보험료 중 일정액을 기준으로 "
 "보험료 및 해약환급금<br>산출방법서에서 정한 방법에 따라 계산한 금액을 말합니<br>다.</p><p id='38' "
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
 'indexing': {'chunk_id': 'chunk_000175',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
