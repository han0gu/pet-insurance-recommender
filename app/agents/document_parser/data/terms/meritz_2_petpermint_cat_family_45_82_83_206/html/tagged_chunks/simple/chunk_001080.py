from langchain_core.documents import Document

chunk = Document(
    page_content=('때<br>다) 폐질환 또는 폐 부분절제술 후 일상생활에서 호<br>흡곤란으로 지속적인 산소치료가 필요하며, 폐기<br>능 검사(PFT)상 '
 "폐환기 기능(1초간 노력성 호기<br>량, FEV1)이 정상예측치의 40% 이하로 저하된 때</p><br><p id='29' "
 "data-category='list' style='font-size:20px'>6) 흉복부, 비뇨생식기계 장해는 질병 또는 외상의 "
 '직접<br>결과로 인한 장해를 말하며, 노화에 의한 기능장해<br>또는 질병이나 외상이 없는 상태에서 예방적으로 장<br>기를 절제, '
 '적출한'),
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
 'indexing': {'chunk_id': 'chunk_001080',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
