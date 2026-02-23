from langchain_core.documents import Document

chunk = Document(
    page_content=('부활(효력회복))를 따릅니다.<br>이 경우 부활(효력회복)일을 계약일로 하여 제5항 및 제6항<br>의 보장개시일을 '
 "적용합니다.</p><h1 id='87' style='font-size:16px'>② 기본형</h1><br><p id='88' "
 "data-category='paragraph' style='font-size:16px'>\uf000 회사는 보험기간 중에 보험증권에 "
 '기재된 반려동물에게<br>질병 또는 상해가 발생하여 그 치료를 직접적인 목적으로<br>수의사법 제2조(정의)에서 정한 국내 동물병원(이하 '
 '「동물<br>병원」이라'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000764',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
