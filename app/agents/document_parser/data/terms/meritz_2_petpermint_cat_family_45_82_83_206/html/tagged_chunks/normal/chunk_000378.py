from langchain_core.documents import Document

chunk = Document(
    page_content=('것으로 봅니다.<br>\uf000 회사가 청약과 함께 제1회 보험료를 받고 청약을 승낙하<br>기 전에 보험금 지급사유가 발생하였을 '
 "때에도 보장개시일<br>부터 이 약관이 정하는 바에 따라 보장을 합니다.</p><br><h1 id='27' "
 "style='font-size:20px'>【보장개시일】</h1><br><p id='28' data-category='paragraph' "
 "style='font-size:16px'>회사가 보장을 개시하는 날로서 계약이 성립되고 제1회 보<br>험료를 받은 날을 말하나, 회사가 "
 '승낙하기 전이라도 청약<br>과'),
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
 'indexing': {'chunk_id': 'chunk_000378',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
