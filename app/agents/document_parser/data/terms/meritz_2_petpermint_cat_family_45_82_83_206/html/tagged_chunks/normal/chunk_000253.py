from langchain_core.documents import Document

chunk = Document(
    page_content=('원하지<br>않거나, 새로운 보장내용으로 계약내용을 변경하는 것이 불<br>가능한 경우, 회사는 계약자에게 이 계약의「보험료 및 '
 '해<br>약환급금 산출방법서」에서 정하는 바에 따라 계약내용 변<br>경시점의 계약자적립액 및 미경과보험료를 지급하고, 이 계<br>약은 '
 "더 이상 효력을 가지지 않습니다.</p><h1 id='52' style='font-size:20px'>제45조(회사의 "
 "손해배상책임)</h1><br><p id='53' data-category='paragraph' "
 "style='font-size:20px'>\uf000 회사는"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000253',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
