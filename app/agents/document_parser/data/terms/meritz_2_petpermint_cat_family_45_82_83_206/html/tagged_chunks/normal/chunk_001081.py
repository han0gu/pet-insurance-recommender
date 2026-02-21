from langchain_core.documents import Document

chunk = Document(
    page_content=('기능장해<br>또는 질병이나 외상이 없는 상태에서 예방적으로 장<br>기를 절제, 적출한 경우는 장해로 보지 않는다.<br>7) 상기 '
 '흉복부 및 비뇨생식기계 장해항목에 명기되지<br>않은 기타 장해상태에 대해서는 “<붙임>일상생활<br>기본동작(ADLs) 제한 '
 '장해평가표”에 해당하는 장해<br>가 있을 때 ADLs 장해 지급률을 준용한다.<br>8) 상기 장해항목에 해당되지 않는 장기간의 간병이 '
 "필요<br>한 만성질환(만성간질환, 만성폐쇄성폐질환 등)은 장<br>해의 평가 대상으로 인정하지 않는다.</p><p id='30'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_001081',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
