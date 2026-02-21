from langchain_core.documents import Document

chunk = Document(
    page_content=('척추체(척추뼈 몸통)에 골절 또는 탈구로 2개<br>의 척추체(척추뼈 몸통)를 유합(아물어 붙음) 또는 고<br>정한 상태<br>9) '
 "심한 기형이란 다음 중 어느 하나에 해당하는 경우를<br>말한다.</p><br><p id='4' data-category='list' "
 "style='font-size:16px'>가) 척추(등뼈)의 골절 또는 탈구 등으로 35° 이상<br>의 척추전만증(척추가 앞으로 "
 '휘어지는 증상),<br>척추후만증(척추가 뒤로 휘어지는 증상) 또는<br>20° 이상의 척추측만증(척추가 옆으로 휘어지는<br>증상)'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000993',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
