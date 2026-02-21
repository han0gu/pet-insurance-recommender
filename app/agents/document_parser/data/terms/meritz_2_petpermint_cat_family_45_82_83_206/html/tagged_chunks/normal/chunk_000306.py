from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 정<br>당한 사유 없이 이에 동의하지 않을 경우 사실 확인이 끝날<br>때까지 회사는 보험금 지급지연에 따른 이자를 지급하지 '
 "않<br>습니다.</p><br><p id='37' data-category='paragraph' "
 "style='font-size:20px'>\uf000 회사는 제5항의 서면조사에 대한 동의 요청시 조사목적,</p><br><h1 "
 "id='38' style='font-size:20px'>사용처 등을 명시하고 설명합니다.</h1><br><p id='39' "
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
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000306',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
