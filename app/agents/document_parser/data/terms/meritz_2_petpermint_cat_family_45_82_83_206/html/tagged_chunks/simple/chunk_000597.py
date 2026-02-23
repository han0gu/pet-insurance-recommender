from langchain_core.documents import Document

chunk = Document(
    page_content=(". 펫퍼민트 반려묘 입원의료비Ⅱ보장 특별약관</p><h1 id='57' style='font-size:16px'>제1조(보험금의 "
 "지급사유)</h1><br><h1 id='58' style='font-size:18px'>① 고급형</h1><br><p id='59' "
 "data-category='paragraph' style='font-size:16px'>\uf000 회사는 보험기간 중에 보험증권에 "
 '기재된 반려동물에게<br>질병 또는 상해가 발생하여 그 치료를 직접적인 목적으로<br>수의사법 제2조(정의)에서 정한 국내 동물병원(이하'),
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
 'indexing': {'chunk_id': 'chunk_000597',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
