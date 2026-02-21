from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>① 청약서의 기재사항을 변경하고자 할 때 또는 변경이<br>생겼음을 알았을 때<br>② 이 "
 '계약에서 보장하는 위험과 동일한 위험을 보장하는<br>계약을 다른 보험자와 체결하고자 할 때 또는 이와 같<br>은 계약이 있음을 알았을 '
 '때<br>③ 반려동물을 양도할 때<br>④ 위 이외에 위험이 뚜렷이 변경되거나 변경되었음을 알<br>았을 때</p><br><p '
 "id='56' data-category='paragraph' style='font-size:16px'>\uf000 회사는 제1항의 통지로 "
 '인하여'),
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
 'indexing': {'chunk_id': 'chunk_000317',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
