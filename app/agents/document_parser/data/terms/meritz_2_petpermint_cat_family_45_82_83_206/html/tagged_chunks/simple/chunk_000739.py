from langchain_core.documents import Document

chunk = Document(
    page_content=("id='48' data-category='paragraph' style='font-size:20px'>강한 자기장 내에서 반려동물에 "
 "고주파를 전사해서 반향<br>되는 전자기파를 측정하여 영상을 얻어 질병을 진단하는<br>검사</p><h1 id='49' "
 "style='font-size:16px'>【전산화단층영상(CT)】</h1><br><p id='50' "
 "data-category='paragraph' style='font-size:20px'>X선을 이용하여 반려동물의 횡단면상의 영상을 "
 '획득하여<br>진단에 이용하는 검사</p><h1'),
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
 'indexing': {'chunk_id': 'chunk_000739',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
