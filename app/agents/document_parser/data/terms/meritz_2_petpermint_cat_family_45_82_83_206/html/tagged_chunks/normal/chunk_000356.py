from langchain_core.documents import Document

chunk = Document(
    page_content=("제1항에 따른 해약환급금을 계<br>약자에게 지급합니다.</p><br><h1 id='1' "
 "style='font-size:20px'>【감액】</h1><br><p id='2' data-category='paragraph' "
 "style='font-size:20px'>보험료, 보험금, 계약자적립액 등을 산정하는 기준이 되<br>는 가입금액을 계약시 선택한 "
 '금액보다 적은 금액으로<br>줄이는 것을 말합니다.(이에 따라 보험료, 보험금 및 해<br>약환급금도 줄어듭니다)</p><br><p '
 "id='3'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000356',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
