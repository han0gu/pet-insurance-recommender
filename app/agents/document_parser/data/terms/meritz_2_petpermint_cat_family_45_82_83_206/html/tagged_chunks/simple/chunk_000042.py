from langchain_core.documents import Document

chunk = Document(
    page_content=('청구)에서 정한 서류를 접수한<br>때에는 접수증을 드리고 휴대전화 문자메시지 또는 전자우<br>편 등으로도 송부하며, 그 서류를 접수한 '
 "날부터 3영업일<br>이내에 보험금을 지급합니다.</p><br><p id='58' data-category='paragraph' "
 "style='font-size:20px'>\uf000 회사가 보험금 지급사유를 조사ㆍ확인하기 위해 필요한</p><footer "
 "id='59' style='font-size:14px'>52</footer><p id='60' "
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
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000042',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
